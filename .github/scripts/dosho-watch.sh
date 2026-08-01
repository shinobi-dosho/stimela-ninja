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
simms_note=""
if command -v uv >/dev/null; then
    cd "$ROOT/stimela-ninja"
    # Guarded, because `set -e` is in force and this script's whole job is to
    # report. An unguarded failure here would abort before the issue is
    # written, so the watch would go *silent* on exactly the days something
    # was wrong -- the failure mode this script exists to prevent.
    if ! { uv sync --quiet --python 3.12 --group dev \
        && uv pip install --quiet "$ROOT/dosho"; } 2>"$RUNNER_TEMP/setup.err"; then
        ci_status=fail
        ci_detail="environment setup failed before tests ran:
$(cat "$RUNNER_TEMP/setup.err")"
        cd "$ROOT/dosho"
    fi
fi

if [[ "$ci_status" == skip ]] && command -v uv >/dev/null; then
    cd "$ROOT/stimela-ninja"

    # simms, installed separately and tolerantly. dosho's skysim/telsim are
    # `@shinobi.pystep` StepRefs, not Cabs, so a backend override cannot
    # intercept them -- they run their own body and `import_func` the real
    # `simms.apps.*` at execution time. Without simms present the one test
    # that dispatches them skips, and dosho's pysteps go unexercised in the
    # only place that tests them against dosho main.
    #
    # Not --no-deps: the simms app modules import dask/daskms/numpy at module
    # level, so a dependency-less install fails the same way one module later.
    #
    # Failure here is reported, never fatal. simms is an unreleased git
    # dependency; an upstream breakage there is not stimela-ninja breaking,
    # and folding it into ci_status would say it was.
    if uv pip install --quiet "simms @ git+https://github.com/wits-cfa/simms.git" 2>/dev/null; then
        simms_note="simms installed from git -- dosho's skysim/telsim pysteps ran for real."
    else
        simms_note="⚠️ simms could not be installed, so dosho's skysim/telsim pysteps were **skipped**, not tested."
    fi

    # simms depends on stimela-ninja, so installing it resolves that dependency
    # and drags a PyPI *wheel* over the editable checkout -- silently switching
    # the whole run to testing a released stimela-ninja against dosho tip, which
    # is not what this job is for. Re-assert the checkout; --no-deps so nothing
    # else is re-resolved on the way past.
    uv pip install --quiet --no-deps -e . 2>/dev/null || true

    # Then verify it, because that failure is invisible on its own: the suite
    # still runs, still passes or fails plausibly, and reports on a tree nobody
    # is looking at. One import is cheap; a week of confident results about the
    # wrong code is not.
    want=$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$PWD/src/shinobi")
    got=$(uv run --no-sync python -c \
        'import shinobi,os;print(os.path.realpath(os.path.dirname(shinobi.__file__)))' 2>/dev/null || true)
    if [[ "$got" != "$want" ]]; then
        ci_status=fail
        ci_detail="environment is not testing this checkout: shinobi imports from
${got:-<unimportable>}
expected
$want
Something in the install chain replaced the editable project (simms and dosho
both depend on stimela-ninja). Tests were not run."
        cd "$ROOT/dosho"
    else
        ci_status=pass
        ci_detail=$(uv run --no-sync pytest -q 2>&1) || ci_status=fail
        cd "$ROOT/dosho"
    fi
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
    prev_failing=no; [[ "$prev_title" == *"CI FAILING"* ]] && prev_failing=yes
    now_failing=no;  [[ "$ci_status" == fail ]] && now_failing=yes
    [[ "$prev_failing" == "$now_failing" ]] || prev_num=""
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
    gh issue create --repo "$REPO" --label dosho-watch \
        --title "$title" \
        --body-file "$body"
fi
