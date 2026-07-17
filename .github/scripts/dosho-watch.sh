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

# New activity: run stimela-ninja's CI tests (same invocation as ci.yml,
# --group dev) with dosho installed from today's main on top of the synced
# environment. --no-sync on the pytest run keeps uv from reverting dosho
# to the locked commit.
ci_status=skip
ci_detail="uv not on PATH; tests not run"
if command -v uv >/dev/null; then
    ci_status=pass
    ci_detail=$({ cd "$ROOT/stimela-ninja" \
        && uv sync --quiet --python 3.12 --group dev \
        && uv pip install --quiet "$ROOT/dosho" \
        && uv run --no-sync pytest -q; } 2>&1) || ci_status=fail
fi

body="$RUNNER_TEMP/dosho-watch-body.md"
{
    echo "Automated digest of new activity on shinobi-dosho/dosho main," \
         "classified by conventional-commit markers. Review the commits" \
         "below alongside the CI result."
    echo
    echo "## CI tests (\`--group dev\` + dosho@main)"
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
gh issue create --repo "$REPO" --label dosho-watch \
    --title "$title" \
    --body-file "$body"
