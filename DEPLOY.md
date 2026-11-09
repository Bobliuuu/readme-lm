# Deploy README-LM to your own GitHub profile

Result: a tiny language model that lives in **your profile README**. Anyone opens an issue, an Action runs the model on their text, and the animation at the top of your profile changes. About 5 minutes, no servers.

## Quickstart (5 minutes)

Prerequisites: [`gh`](https://cli.github.com) logged in (`gh auth login`, token scopes `repo` and `workflow`), `git`, `python3` with `numpy` (`pip install numpy`).

```bash
gh repo fork Bobliuuu/readme-lm --clone && cd readme-lm   # 1. fork + clone
scripts/deploy.sh --dry-run                               # 2. optional: print everything it would do
scripts/deploy.sh                                         # 3. deploy (asks before every write)
```

4. Open one of the links it prints (for example `https://github.com/YOU/readme-lm/issues/new?title=qa%3A+example+text`), press **Submit new issue**, and watch `https://github.com/YOU/YOU` update in about a minute. `--test` does this step for you.

Flags: `--owner NAME` (default: the account `gh` is logged in as), `--dry-run`, `--yes` (no questions), `--test`, `--rotate-key`. The script is idempotent: run it again any time. It refuses to run if it cannot verify `gh` auth.

## What each piece does

| Piece | Where | Job |
|---|---|---|
| `demos.yml` | `YOU/readme-lm/.github/workflows` | Runs on issues titled `gpt2:`, `qa:`, `mediawiki:` or `svg:`; computes the result, rewrites `assets/readme-lm.svg` in your profile repo, replies on the issue and closes it. Profile repo is always `YOU/YOU` (no editing needed). |
| Issue templates | `.github/ISSUE_TEMPLATE` | Pre-fill the title (`qa: example text`). |
| Profile repo | `YOU/YOU` | Holds your README (with the model weights between `<!--LM-WEIGHTS-START/END-->`), `readme-lm/*.py`, the animation, and a cron-only `readme-lm.yml`. Built from `profile-template/` by `scripts/build_profile.py`. |
| Deploy key + secret | profile repo + `PROFILE_DEPLOY_KEY` on the fork | Lets the fork's Action push to the profile repo, and nothing else. |

## Three versions

- **A. GPT-2 / QA on a minimal Action.** `gpt2:` runs real GPT-2 small (from the Markdown files in `mdrt/`), `qa:` runs the 600K-parameter QA model stored in your README. The Action runs numpy.
- **B. Pure MediaWiki.** `demos/mediawiki/README.mediawiki` is a page whose arithmetic GitHub's own MediaWiki renderer computes. You can just open the page, no Action needed. With the Action, it only writes your typed text into the page, reads the rendered result back and animates your profile.
- **C. Pure SVG.** `demos/svg/latest.svg` is a filter network computed by Chrome's filter engine when you open the SVG. Same story: the Action only types your text into the SVG and reads the result back.

## Customize the QA facts

The QA model knows what is in `qa/g*.jsonl` (question/answer pairs, currently about the original author's repos).

1. Edit or replace the `qa/g*.jsonl` files with facts about you.
2. Retrain: `python3 train_qa.py EPOCHS HOLDOUT TAG [d L ff]` (it reads `qa/g*.jsonl` through `build_qa.py`; a GPU or Apple MPS helps). Output is a `qa_TAG.pt` checkpoint.
3. Repack into a README: `python3 export_qa.py MODEL.pt /path/to/YOU/README.md` (writes the 4-bit weights between the `LM-WEIGHTS` markers). To refresh `qa/weights.md` for future deploys, copy the base64 block from that README into it.
4. Commit and push the profile repo. Edit the preset questions in `ask.py` (`PRESETS`) and push the copies in `readme-lm/` too.

## Uninstall

```bash
scripts/undeploy.sh            # removes the deploy key and the secret (asks first); --dry-run to preview
gh workflow disable demos.yml -R YOU/readme-lm
```
Then delete the `LM-DEMO` and `LM-WEIGHTS` sections from your profile README, and `readme-lm/`, `assets/readme-lm.svg` and `.github/workflows/readme-lm.yml` from the profile repo. Delete the fork if you like.

## Troubleshooting

- **Issue form opens blank.** Type the title yourself: `qa: what is this`, `gpt2: once upon a time`, `mediawiki: hello`, `svg: hello w`.
- **Nothing happens after submitting.** Is Actions enabled on the fork (Actions tab, enable workflows)? Are Issues enabled? Titles must start with the exact prefix.
- **Run fails at "check out the profile repo".** Deploy key missing or wrong: `scripts/deploy.sh --rotate-key`. The profile repo must be named exactly `YOU/YOU`; the fork must have the secret `PROFILE_DEPLOY_KEY`.
- **Push rejected in the Action.** The deploy key must be created with write access (the script does this); check Profile repo > Settings > Deploy keys.
- **SVG demo looks wrong.** It is only correct at 100% zoom on a non-Retina display in a Chrome-based browser. The Action's headless Chrome is always right, so the issue reply and profile animation are correct either way.
- **README too large.** GitHub renders READMEs up to 500 KiB. The model block is about 410 KB; keep your own text under ~80 KB. `build_profile.py` refuses to go over.
- **Actions minutes.** Public repos get free minutes. A QA run takes about a minute, GPT-2 a few minutes (it installs numpy/regex). The cron preset runs every 3 hours; remove the `schedule` from the profile `readme-lm.yml` to stop it.

## Security notes

- The deploy key is scoped to **one repo** (your profile repo) and can only push to it. It is stored as a secret on the fork; the script never prints it and deletes its local copies. Revoke any time with `undeploy.sh`.
- Issue titles are **untrusted input**: anyone can open an issue on a public repo. The workflow passes them through environment variables (never interpolated into shell), strips newlines and backticks, limits them to 200 characters, replaces `@` so nobody is pinged, and the models only ever see plain text.
- Consequence: strangers can change your profile animation (and trigger minutes). If you do not want that, turn on "Limit to collaborators" for Issues, or run `workflow_dispatch` manually only.
- Forked Actions never receive your secrets for pull requests from other forks.

## Things to change (the original author's own links)

These stay as they are in the repo; replace them if you publish your own copy:
- `README.md` (project README, links to `Bobliuuu/...`), `graph/make_profile_menu.py` (generates the author's no-Action menu page; link to `Bobliuuu/readme-lm`).
- `qa/g*.jsonl`, `big.json`, `repos.json`, `trackD/**` and `demos/**/REPORT.md` (facts and reports about the author's repos).
- `scripts/deploy.sh`: `UPSTREAM` (where forks come from; override with `README_LM_UPSTREAM`).
- The profile README bio, hero and cards (never touched by the deploy script).
