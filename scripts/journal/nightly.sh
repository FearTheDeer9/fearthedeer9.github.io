#!/bin/zsh
# Nightly: sync the Notion daily log, score every day since lock-in, fold new reading into the
# reading log, and publish the public data files to main (which deploys the site).
# Run by ~/Library/LaunchAgents/com.harel.site-nightly.plist; safe to run by hand.
set -euo pipefail
cd "${0:A:h}/../.."
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin

fail() { osascript -e "display notification \"$1\" with title \"Site nightly failed\"" || true; echo "FAILED: $1"; exit 1; }
trap 'fail "step failed at line $LINENO; see private/journal/nightly.log"' ERR

# Pushes go out as the FearTheDeer9 account, whatever account gh has active; the token stays out of argv.
git_gh() { git -c credential.helper= -c 'credential.helper=!f() { echo username=FearTheDeer9; echo "password=$(gh auth token --user FearTheDeer9)"; }; f' "$@"; }

echo "=== $(date '+%F %T')"
[[ "$(git branch --show-current)" == main ]] || fail "repo is not on main"
git_gh pull --ff-only -q

node scripts/journal/sync-notion.mjs
node scripts/journal/collect.mjs
node scripts/journal/score.mjs
node scripts/journal/reading-from-journal.mjs
node scripts/build-reading.mjs

npm run build --silent >/dev/null
node scripts/check-links.mjs
node scripts/check-private.mjs

[[ -n "${NO_PUSH:-}" ]] && { git status --short src/data; echo "NO_PUSH set: not committing"; exit 0; }
git add src/data/scores.json src/data/scorecard.json src/data/reading.json
if git diff --cached --quiet; then echo "nothing new"; exit 0; fi
# Unsigned on purpose: GPG would wait on a passphrase prompt that nobody answers at night.
git -c commit.gpgsign=false -c user.name='FearTheDeer9 (nightly)' -c user.email='39518454+FearTheDeer9@users.noreply.github.com' \
  commit -q -m "Nightly: scores and reading through $(date +%F)"
git_gh push -q
echo "pushed $(git rev-parse --short HEAD)"
