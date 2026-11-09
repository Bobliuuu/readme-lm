#!/usr/bin/env bash
# undeploy.sh: remove the deploy key (from OWNER/OWNER) and the secret PROFILE_DEPLOY_KEY (from the fork). Asks first.
# Does not delete repos or revert README edits (see DEPLOY.md > Uninstall).
set -euo pipefail
OWNER=""; REPO="readme-lm"; DRY=0; YES=0
while [ $# -gt 0 ]; do case "$1" in
  --owner) OWNER="${2:-}"; shift 2;; --repo) REPO="${2:-}"; shift 2;; --dry-run) DRY=1; shift;; --yes|-y) YES=1; shift;;
  -h|--help) echo "usage: scripts/undeploy.sh [--owner NAME] [--repo readme-lm] [--dry-run] [--yes]"; exit 0;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
run() { printf '+ %s\n' "$*"; if [ "$DRY" = 1 ]; then return 0; fi; "$@"; }
command -v gh >/dev/null || die "gh not found"
gh auth status >/dev/null 2>&1 || die "gh is not logged in (cannot verify auth, refusing)"
ME="$(gh api user -q .login 2>/dev/null)" || die "could not read your GitHub login"; OWNER="${OWNER:-$ME}"
PROFILE="$OWNER/$OWNER"; FORK="$OWNER/$REPO"
echo "will remove: deploy key 'readme-lm-actions' from $PROFILE and secret PROFILE_DEPLOY_KEY from $FORK"
if [ "$DRY" = 0 ] && [ "$YES" = 0 ]; then
  [ -r /dev/tty ] || die "no terminal; use --yes"; printf 'Proceed? [y/N] ' > /dev/tty; read -r a < /dev/tty; case "$a" in y|Y|yes) ;; *) die "aborted";; esac
fi
KID="$(gh repo deploy-key list -R "$PROFILE" 2>/dev/null | awk '$2=="readme-lm-actions"{print $1; exit}')" || true
if [ -n "${KID:-}" ]; then run gh repo deploy-key delete "$KID" -R "$PROFILE"; else echo "no deploy key found on $PROFILE"; fi
if gh secret list -R "$FORK" 2>/dev/null | grep -q '^PROFILE_DEPLOY_KEY'; then run gh secret delete PROFILE_DEPLOY_KEY -R "$FORK"; else echo "no secret found on $FORK"; fi
echo "done. Issues in $FORK will now fail at the profile checkout step; disable the workflow too if you want it quiet: gh workflow disable demos.yml -R $FORK"
