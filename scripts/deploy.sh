#!/usr/bin/env bash
# deploy.sh: put README-LM on your own GitHub account (fork + profile repo + deploy key + secret).
# Idempotent. Every write asks first (unless --yes). --dry-run prints each write command and runs none.
set -euo pipefail

UPSTREAM="${README_LM_UPSTREAM:-Bobliuuu/readme-lm}"   # where forks come from
KEY_TITLE="readme-lm-actions"; SECRET="PROFILE_DEPLOY_KEY"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=0; YES=0; TEST=0; ROTATE=0; OWNER=""

usage() { cat <<'U'
usage: scripts/deploy.sh [--owner NAME] [--dry-run] [--yes] [--test] [--rotate-key]
  --owner NAME   GitHub user/org to deploy to (default: the account gh is logged in as)
  --dry-run      print every write command, run none (reads and local temp files only)
  --yes          do not ask before writes (still refuses if gh auth cannot be verified)
  --test         after deploying, open a `qa: example text` issue to try it (asks first)
  --rotate-key   replace the deploy key and secret even if both already exist
U
}
while [ $# -gt 0 ]; do
  case "$1" in
    --owner) OWNER="${2:-}"; [ -n "$OWNER" ] || { echo "--owner needs a name" >&2; exit 2; }; shift 2;;
    --dry-run) DRY=1; shift;; --yes|-y) YES=1; shift;; --test) TEST=1; shift;; --rotate-key) ROTATE=1; shift;;
    -h|--help) usage; exit 0;; *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

say()  { printf '%s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }
# run: print the command; execute it unless --dry-run (use ONLY for writes)
run()  { printf '+ %s\n' "$*"; if [ "$DRY" = 1 ]; then return 0; fi; "$@"; }
confirm() {
  if [ "$DRY" = 1 ]; then say "[dry-run] would ask: $1 [y/N]"; return 0; fi
  if [ "$YES" = 1 ]; then say "[--yes] $1"; return 0; fi
  [ -r /dev/tty ] || die "no terminal to ask '$1' (re-run with --yes to skip questions)"
  printf '%s [y/N] ' "$1" > /dev/tty; local a; read -r a < /dev/tty; case "$a" in y|Y|yes|YES) return 0;; *) return 1;; esac
}

TMP="$(mktemp -d)"; chmod 700 "$TMP"
cleanup() {
  if [ -d "$TMP/key" ]; then find "$TMP/key" -type f -exec sh -c 'shred -u "$1" 2>/dev/null || rm -f "$1"' _ {} \; ; fi
  rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

step "checks"
for c in gh git python3; do command -v "$c" >/dev/null || die "$c not found on PATH"; done
python3 -c 'import numpy' 2>/dev/null || die "python3 needs numpy (pip install numpy)"
command -v ssh-keygen >/dev/null || die "ssh-keygen not found"
gh auth status >/dev/null 2>&1 || die "gh is not logged in; run: gh auth login  (cannot verify auth, refusing to continue)"
ME="$(gh api user -q .login 2>/dev/null)" || die "could not read your GitHub login through gh (auth problem?), refusing to continue"
[ -n "$ME" ] || die "empty GitHub login, refusing to continue"
SCOPES="$(gh api -i user 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="x-oauth-scopes"{print $2}')" || true
if [ -n "$SCOPES" ]; then
  for s in repo workflow; do
    case ",$(printf '%s' "$SCOPES" | tr -d ' ')," in *",$s,"*) ;; *) die "gh token lacks scope '$s'. Run: gh auth refresh -s repo,workflow";; esac
  done
  say "gh logged in as $ME with scopes: $SCOPES"
else
  say "gh logged in as $ME (token scopes could not be read; fine-grained token?)"
  confirm "Continue without verifying the repo and workflow scopes?" || die "aborted"
fi
OWNER="${OWNER:-$ME}"
if [ "$OWNER" != "$ME" ]; then say "note: deploying to '$OWNER' while logged in as '$ME' needs admin rights there."; fi
PROFILE="$OWNER/$OWNER"

step "plan"
ORIGIN="$(git -C "$ROOT" remote get-url origin 2>/dev/null | sed -E 's#^.*github\.com[:/]##; s#\.git$##' || true)"
if [ -n "$ORIGIN" ] && [ "$(printf '%s' "${ORIGIN%%/*}" | tr A-Z a-z)" = "$(printf '%s' "$OWNER" | tr A-Z a-z)" ]; then
  FORK="$ORIGIN"; NEED_FORK=0
else
  NEED_FORK=1; FORK="$OWNER/${UPSTREAM#*/}"
  if gh repo view "$FORK" >/dev/null 2>&1; then NEED_FORK=0; fi
fi
say "engine repo (the fork) : $FORK$([ "$NEED_FORK" = 1 ] && echo '  (will be forked from '"$UPSTREAM"')')"
say "profile repo          : $PROFILE"
say "deploy key / secret   : $KEY_TITLE on $PROFILE (write) / $SECRET on $FORK"
[ "$DRY" = 1 ] && say "mode                  : DRY RUN, nothing will be written to GitHub"
confirm "Proceed?" || die "aborted"
REPO="${FORK#*/}"

step "1/6 fork"
if [ "$NEED_FORK" = 1 ]; then
  confirm "Fork $UPSTREAM into $OWNER?" || die "aborted"
  if [ "$OWNER" = "$ME" ]; then run gh repo fork "$UPSTREAM" --clone=false; else run gh repo fork "$UPSTREAM" --clone=false --org "$OWNER"; fi
else say "already have $FORK, nothing to do"; fi

step "2/6 profile repo $PROFILE"
PROFILE_EXISTS=1
if ! gh repo view "$PROFILE" >/dev/null 2>&1; then
  PROFILE_EXISTS=0
  confirm "Create public repo $PROFILE (GitHub shows its README on your profile)?" || die "aborted"
  run gh repo create "$PROFILE" --public --description "Profile README"
else say "$PROFILE exists"; fi
HAS_ISSUES="$(gh repo view "$FORK" --json hasIssuesEnabled -q .hasIssuesEnabled 2>/dev/null || echo false)"
if [ "$HAS_ISSUES" != "true" ]; then
  confirm "Enable Issues on $FORK (the demos are triggered by issues)?" || die "aborted"
  run gh repo edit "$FORK" --enable-issues
else say "Issues already enabled on $FORK"; fi

step "3/6 populate $PROFILE from profile-template/"
PDIR="$TMP/profile"
if [ "$PROFILE_EXISTS" = 1 ]; then gh repo clone "$PROFILE" "$PDIR" -- -q 2>/dev/null || { mkdir -p "$PDIR"; git -C "$PDIR" init -q -b main; }
else mkdir -p "$PDIR"; git -C "$PDIR" init -q -b main; fi
if [ -f "$PDIR/README.md" ]; then
  say "$PROFILE already has a README.md: only the model block and the demo section will be added or refreshed, the rest is untouched."
  confirm "Inject the README-LM sections into your existing README?" || die "aborted"
fi
python3 -B "$ROOT/scripts/build_profile.py" "$PDIR" --owner "$OWNER" --repo "$REPO"
git -C "$PDIR" add -A
if git -C "$PDIR" diff --cached --quiet 2>/dev/null; then say "profile repo already up to date"
else
  git -C "$PDIR" --no-pager diff --cached --stat | grep -v 'README.md' || true
  confirm "Commit and push these changes to $PROFILE?" || die "aborted"
  if [ "$DRY" = 1 ]; then
    say "+ git -C <clone of $PROFILE> commit -m 'feat: add README-LM demos'"; say "+ git -C <clone of $PROFILE> push origin HEAD"
  else
    git -C "$PDIR" -c user.name="${GIT_AUTHOR_NAME:-readme-lm deploy}" -c user.email="${GIT_AUTHOR_EMAIL:-readme-lm@users.noreply.github.com}" commit -q -m "feat: add README-LM demos"
    printf '+ git push origin HEAD\n'
    git -C "$PDIR" remote get-url origin >/dev/null 2>&1 || git -C "$PDIR" remote add origin "https://github.com/$PROFILE.git"
    git -C "$PDIR" push origin HEAD
  fi
fi

step "4/6 deploy key (write access to $PROFILE only) and secret $SECRET"
HAVE_KEY=0; HAVE_SECRET=0
gh repo deploy-key list -R "$PROFILE" 2>/dev/null | grep -q "$KEY_TITLE" && HAVE_KEY=1
gh secret list -R "$FORK" 2>/dev/null | grep -q "^$SECRET" && HAVE_SECRET=1
if [ "$HAVE_KEY" = 1 ] && [ "$HAVE_SECRET" = 1 ] && [ "$ROTATE" = 0 ]; then
  say "deploy key and secret already present (use --rotate-key to replace)"
else
  confirm "Generate an ed25519 key, add its public half to $PROFILE with write access, store the private half as secret $SECRET on $FORK?" || die "aborted"
  if [ "$HAVE_KEY" = 1 ]; then
    KID="$(gh repo deploy-key list -R "$PROFILE" | awk -v t="$KEY_TITLE" '$2==t{print $1; exit}')"
    [ -n "$KID" ] && run gh repo deploy-key delete "$KID" -R "$PROFILE"
  fi
  mkdir -p "$TMP/key"; chmod 700 "$TMP/key"
  if [ "$DRY" = 1 ]; then
    say "+ ssh-keygen -t ed25519 -N '' -C $KEY_TITLE -f <tmp>/key/id   (private key never printed)"
    say "+ gh repo deploy-key add <tmp>/key/id.pub -R $PROFILE -t $KEY_TITLE -w"
    say "+ gh secret set $SECRET -R $FORK < <tmp>/key/id"
    say "+ shred -u <tmp>/key/id  (or rm)"
  else
    ssh-keygen -q -t ed25519 -N '' -C "$KEY_TITLE" -f "$TMP/key/id"
    gh repo deploy-key add "$TMP/key/id.pub" -R "$PROFILE" -t "$KEY_TITLE" -w
    gh secret set "$SECRET" -R "$FORK" < "$TMP/key/id" >/dev/null
    say "secret $SECRET stored on $FORK; local key copies are being deleted"
    cleanup_key() { find "$TMP/key" -type f -exec sh -c 'shred -u "$1" 2>/dev/null || rm -f "$1"' _ {} \; ; }; cleanup_key
  fi
fi

step "5/6 enable Actions on $FORK"
if [ "$DRY" = 1 ]; then
  say "+ gh api -X PUT repos/$FORK/actions/permissions -F enabled=true -f allowed_actions=all"
  say "+ gh workflow enable demos.yml -R $FORK"
else
  gh api -X PUT "repos/$FORK/actions/permissions" -F enabled=true -f allowed_actions=all >/dev/null 2>&1 || say "could not set Actions permissions through the API; enable Actions in $FORK > Settings > Actions"
  gh workflow enable demos.yml -R "$FORK" >/dev/null 2>&1 || say "(workflow demos.yml already enabled, or enable it in the Actions tab of $FORK)"
fi

step "6/6 done"
say "Open one of these, keep or edit the title text, and press Submit (the animation on https://github.com/$PROFILE updates in about a minute):"
for t in qa gpt2 mediawiki svg; do say "  $t: https://github.com/$FORK/issues/new?title=$t%3A+example+text"; done
say "If the form opens blank, create an issue titled like 'qa: what is this'."
if [ "$TEST" = 1 ]; then
  if confirm "Open a test issue 'qa: example text' on $FORK now?"; then
    run gh issue create -R "$FORK" --title "qa: example text" --body "Test issue from scripts/deploy.sh (the Action will answer and close it)."
  fi
fi
