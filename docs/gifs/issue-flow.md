# issue-flow: README-LM issue flow, screen recording

`issue-flow.mp4` is an uncut 90 s screen recording (macOS `screencapture -v`, cropped to the Chrome window) of one live run on 2026-11-09 (issue Bobliuuu/readme-lm#15, `qa: what is jerry building`). `issue-flow.gif` is the same recording at 6 fps, 480 px wide.

What it shows, in order: the README `qa:` link is clicked, the new-issue form opens prefilled, the title is edited, Create is clicked, the issue page appears, the bot comment arrives (about 20 s after Create) and the issue is closed.

Notes:
- The answer is what the model really produced; it is a real answer, not a good one.
- The recording ends at the closed issue; the profile animation refresh is not in it (GitHub's raw-file cache can delay that by minutes).
- `issue-flow-frames/` holds the screenshots of the earlier run (issue #12) that the explainer's live-demo scene still uses.
