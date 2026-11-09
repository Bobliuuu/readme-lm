## Deploy it yourself

Run README-LM on your own GitHub profile in about 5 minutes. Needs the [`gh` CLI](https://cli.github.com) (logged in), `git`, and `python3` with `numpy`.

```bash
gh repo fork Bobliuuu/readme-lm --clone && cd readme-lm
scripts/deploy.sh --dry-run   # preview every command, writes nothing
scripts/deploy.sh             # forks, sets up YOU/YOU, deploy key, secret; asks before each write
```

Then open the issue link it prints (`qa:`, `gpt2:`, `mediawiki:` or `svg:`) and your profile animation updates in about a minute. Full guide, customization and troubleshooting: [DEPLOY.md](DEPLOY.md). Remove it again with `scripts/undeploy.sh`.
