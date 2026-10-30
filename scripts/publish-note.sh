#!/bin/bash
# Publish the note open in Typora: commit it and push to GitHub. Cloudflare
# Pages (project "voidindex-site") builds every push to main, so the note is
# live at voidindex.net a minute or two later. Typora runs this from
# Export > "Publish to VoidIndex" with the current file's path as the first
# argument.
#
#   scripts/publish-note.sh <path-to-note.md>

# Typora starts commands with a minimal PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

LOG="$HOME/Library/Logs/voidindex-publish.log"
NOTE="$1"

notify() {
  osascript -e "display notification \"$2\" with title \"VoidIndex\" subtitle \"$1\"" >/dev/null 2>&1
  echo "$(date '+%F %T') $1: $2" >>"$LOG"
}

fail() {
  notify "Publish failed" "$1"
  exit 1
}

[ -f "$NOTE" ] || fail "No file to publish (got: $NOTE)"

REPO="$(git -C "$(dirname "$NOTE")" rev-parse --show-toplevel 2>/dev/null)" ||
  fail "$(basename "$NOTE") is not inside a git repository"

case "$NOTE" in
  "$REPO"/src/content/posts/*) ;;
  *) fail "$(basename "$NOTE") is not in src/content/posts" ;;
esac

cd "$REPO" || fail "Cannot open $REPO"
REL="${NOTE#"$REPO"/}"
TITLE="$(sed -n 's/^title: *//p' "$NOTE" | head -1 | tr -d '"')"
TITLE="${TITLE:-$(basename "$NOTE" .md)}"

if grep -q '^draft: *true' "$NOTE"; then
  notify "Not published" "$TITLE is a draft (draft: true)"
  exit 0
fi

git add -- "$REL" >>"$LOG" 2>&1 || fail "git add failed"
if git diff --cached --quiet -- "$REL"; then
  notify "Nothing to publish" "$TITLE has no new changes"
  exit 0
fi

git commit -q -m "Publish: $TITLE" -- "$REL" >>"$LOG" 2>&1 || fail "git commit failed"
git pull -q --rebase --autostash origin main >>"$LOG" 2>&1 || fail "git pull failed (see $LOG)"
git push -q origin HEAD:main >>"$LOG" 2>&1 || fail "git push failed (see $LOG)"

notify "Pushed" "$TITLE will be live on voidindex.net in about 2 minutes"
