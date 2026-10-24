#!/usr/bin/env bash
# Daily dosho-watch digest. Classifies new commits on dosho main by
# conventional-commit markers (type!: subjects, BREAKING CHANGE footers,
# feat:) plus new tags/releases, and opens an issue on stimela-ninja.
# The issue body ends with a "Last checked dosho commit: <sha>" marker
# that the next run reads to know where to resume.
#
# When there is new activity, it also runs stimela-ninja's CI tests
# (uv, --group dev) with dosho installed from the fresh clone, so obvious
# breakage shows up in the issue instead of waiting to be discovered.
#
# Expects: cwd containing ./stimela-ninja (this repo) and ./dosho (clone of
# shinobi-dosho/dosho, main checked out), uv on PATH, and GH_TOKEN with
# issues:write on stimela-ninja.
set -euo pipefail

REPO=shinobi-dosho/stimela-ninja
DOSHO=shinobi-dosho/dosho

ROOT=$PWD
watch_cache="$ROOT/stimela-ninja/.work-cache/dosho-watch"
mkdir -p "$watch_cache/tmp"
export TMPDIR="$watch_cache/tmp"
export UV_CACHE_DIR="$watch_cache/cache"
ninja_sha=$(git -C "$ROOT/stimela-ninja" rev-parse HEAD)
dosho_sha=$(git -C "$ROOT/dosho" rev-parse HEAD)
cd dosho

last_sha=$(gh issue list --repo "$REPO" --label dosho-watch --state all --limit 1 --json body \
    --jq '.[0].body // ""' | grep -oE 'Last checked dosho commit: [0-9a-f]{7,40}' | awk '{print $NF}' || true)

if [[ -n "$last_sha" ]] && git cat-file -e "$last_sha^{commit}" 2>/dev/null; then
    log_cmd=(git log "$last_sha..HEAD")
    since=$(git log -1 --format=%cI "$last_sha")
else
    # No marker issue yet, or dosho history was rewritten: bootstrap window.
    since=$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)
    log_cmd=(git log --since="$since")
fi
since_epoch=$(date -d "$since" +%s)
since_utc=$(date -u -d "$since" +%Y-%m-%dT%H:%M:%SZ)

subjects=$("${log_cmd[@]}" --format='%h %s')

# Breaking: "type(scope)!:" subjects plus commits with a BREAKING CHANGE footer.
breaking=$(printf '%s\n%s\n' \
    "$(printf '%s\n' "$subjects" | grep -E '^[0-9a-f]+ [a-zA-Z]+(\([^)]*\))?!:' || true)" \
    "$("${log_cmd[@]}" --grep='BREAKING[ -]CHANGE' --format='%h %s')" \
    | grep -v '^$' | sort -u || true)
features=$(printf '%s\n' "$subjects" | grep -E '^[0-9a-f]+ feat(\([^)]*\))?[:!]' || true)
routine=$(printf '%s\n' "$subjects" \
    | grep -vE '^[0-9a-f]+ (feat(\([^)]*\))?[:!]|[a-zA-Z]+(\([^)]*\))?!:)' || true)

releases=$(gh api "repos/$DOSHO/releases" --jq \
    ".[] | select(.published_at > \"$since_utc\") | \"- \(.tag_name): \(.name // \"(unnamed)\") \(.html_url)\"" || true)
tags=$(git tag --format='%(creatordate:unix) %(refname:short)' \
    | awk -v s="$since_epoch" '$1 > s {print "- " $2}' || true)

if [[ -z "$subjects" && -z "$releases" && -z "$tags" ]]; then
    echo "dosho unchanged since last check"
    exit 0
fi

# New activity: locked setup first, optional tool second, candidate wheels last.
# Guard every setup failure so the existing issue-reporting path still runs.
ci_status=skip
ci_detail="uv not on PATH; tests not run"
simms_note=""
if command -v uv >/dev/null; then
    cd "$ROOT/stimela-ninja"
    if ! uv sync --quiet --locked --python 3.12 --group dev 2>"$watch_cache/setup.err"; then
        ci_status=fail
        ci_detail="environment setup failed before tests ran:
$(cat "$watch_cache/setup.err")"
    else
        # A best-effort external tool, constrained by the locked dependency set.
        # Failure is reported separately; no locked package is upgraded for it.
        if uv export --locked --no-emit-project --no-hashes --group dev \
            --format requirements-txt -o "$watch_cache/constraints.txt" \
            && uv pip install --quiet --constraint "$watch_cache/constraints.txt" "simms>=3.0.2" 2>/dev/null; then
            simms_note="simms installed from PyPI -- the native skysim/telsim example dispatch ran."
        else
            simms_note="⚠️ simms could not be installed, so native skysim/telsim dispatch was **skipped**; only schema/graph coverage ran."
        fi
        if ! bash .github/scripts/install-candidate-pair.sh "$ROOT/dosho" >"$watch_cache/pair.log" 2>&1; then
            ci_status=fail
            ci_detail="candidate wheel-pair setup failed before tests ran:
$(cat "$watch_cache/pair.log")"
        else
            ci_status=pass
            ci_detail=$(uv run --no-sync pytest -q -o pythonpath= 2>&1) || ci_status=fail
        fi
    fi
    cd "$ROOT/dosho"
fi

body="$watch_cache/dosho-watch-body.md"
{
    echo "Automated digest of new activity on shinobi-dosho/dosho main," \
         "classified by conventional-commit markers. Review the commits" \
         "below alongside the CI result."
    echo
    echo "## CI tests (locked dependencies + candidate wheel pair)"
    echo "Tested source HEADs: stimela-ninja=$ninja_sha; dosho=$dosho_sha"
    case "$ci_status" in
        pass) echo "✅ Passed — \`$(printf '%s\n' "$ci_detail" | tail -n1)\`" ;;
        fail)
            echo "❌ **FAILED** — last 60 lines of output:"
            echo '```'
            printf '%s\n' "$ci_detail" | tail -n 60
            echo '```'
            ;;
        *) echo "⚠️ Skipped: $ci_detail" ;;
    esac
    [[ -n "$simms_note" ]] && { echo; echo "$simms_note"; }
    echo
    section() {
        [[ -z "$2" ]] && return 0
        echo "## $1"
        printf '%s\n' "$2" | sed -E "s|^([0-9a-f]+) |- [\`\1\`](https://github.com/$DOSHO/commit/\1) |"
        echo
    }
    section "Possible breaking changes" "$breaking"
    section "New features" "$features"
    if [[ -n "$releases" || -n "$tags" ]]; then
        echo "## New releases / tags"
        [[ -n "$releases" ]] && printf '%s\n' "$releases"
        [[ -n "$tags" ]] && printf '%s\n' "$tags"
        echo
    fi
    section "Other commits" "$routine"
    echo "Last checked dosho commit: $(git rev-parse HEAD)"
} > "$body"

n_commits=$(printf '%s\n' "$subjects" | grep -c . || true)
title="dosho watch $(date -u +%F): $n_commits new commits"
[[ "$ci_status" == fail ]] && title="$title — CI FAILING"
gh label create dosho-watch --repo "$REPO" \
    --color D93F0B --description "automated dosho monitoring" 2>/dev/null || true

# Roll the open issue forward while the verdict is unchanged, rather than
# opening one per day. Six consecutive identical "CI FAILING" issues is how
# this ran before, and it trains you to skim past the seventh -- the signal
# was correct every day and read on none of them. A *changed* verdict is
# news and still gets its own issue; a repeated one is the same news.
prev=$(gh issue list --repo "$REPO" --label dosho-watch --state open --limit 1 \
    --json number,title,createdAt --jq '.[0] | select(.) | "\(.number)\t\(.title)\t\(.createdAt)"' || true)
prev_num=""
if [[ -n "$prev" ]]; then
    IFS=$'\t' read -r prev_num prev_title prev_created <<<"$prev"
    # Only ever treat a plain integer as an issue number. If the query shape
    # changes under us, or jq returns something unexpected, the alternative is
    # PATCHing repos/<repo>/issues/<whatever-that-was> -- better to fall
    # through and open a new issue than to address an endpoint by garbage.
    [[ "$prev_num" =~ ^[0-9]+$ ]] || prev_num=""
    prev_failing=no; [[ "$prev_title" == *"CI FAILING"* ]] && prev_failing=yes
    now_failing=no;  [[ "$ci_status" == fail ]] && now_failing=yes
    if [[ "$prev_failing" != "$now_failing" ]]; then
        superseded=$prev_num   # verdict changed: new issue, and retire this one
        prev_num=""
    fi
fi

if [[ -n "$prev_num" ]]; then
    # Say how long this verdict has stood, so a stale failure reads as stale.
    days=$(( ( $(date -u +%s) - $(date -u -d "$prev_created" +%s) ) / 86400 ))
    { echo "_Rolling update: the CI verdict has been unchanged since" \
           "${prev_created%%T*} (${days}d). Refreshed daily in place;" \
           "a change of verdict opens a new issue._"
      echo
      cat "$body"
    } > "$body.rolled" && mv "$body.rolled" "$body"

    # gh api, not `gh issue edit`: the latter routes through the deprecated
    # Projects (classic) GraphQL surface and can abort before applying the
    # change, reporting an error while leaving the issue untouched.
    gh api -X PATCH "repos/$REPO/issues/$prev_num" \
        -f title="$title" -F body=@"$body" --silent
    echo "updated existing issue #$prev_num (verdict unchanged)"
else
    new_url=$(gh issue create --repo "$REPO" --label dosho-watch \
        --title "$title" \
        --body-file "$body")
    echo "$new_url"

    # A verdict change retires the issue that reported the old one. Leaving it
    # open is how you end up with a standing "CI FAILING" describing something
    # fixed days ago -- the same stale-signal problem the rolling update above
    # exists to avoid, just slower.
    if [[ -n "${superseded:-}" ]]; then
        gh issue comment "$superseded" --repo "$REPO" \
            --body "Superseded by $new_url — the CI verdict changed, so this report no longer describes the current state." || true
        gh issue close "$superseded" --repo "$REPO" || true
        echo "closed superseded issue #$superseded"
    fi
fi
