# issue-flow: README-LM issue flow in a logged-in Chrome

Every frame is a real screenshot taken during one live run on 2026-11-09 (issue
Bobliuuu/readme-lm#12, `qa: what is jerry building`). Nothing is interpolated,
synthetic or reused from an earlier run. Frames are in `issue-flow-frames/`
(`01.jpg` to `16.jpg`, metadata in `frames.json`); the GIF/MP4 burn a caption and
an elapsed-seconds label onto each (hold 1.0-2.4 s per frame).

Steps, in order:
1. Repo page.
2. Scrolled to "Try it now".
3. The README `qa:` link was clicked (a real click); the new-issue form opened with the title `qa: example text` and the body prefilled.
4. Title edited to `qa: what is jerry building` (real typing).
5. Create clicked (the frame was taken right after the click; the form was still on screen).
6. Issue #12 page right after creation.
7. Waiting (issue page, no reply yet), then the Actions tab showing the workflow run "In progress", then more waiting.
8. Bot comment posted and the issue closed as completed (about 35 s after Create; the run is typically 20-25 s plus page reload lag).
9. The bot's answer in detail.
10. Profile page after a hard reload: the animation image still showed the previous answer (browser/CDN cache, max-age 300 s).
11. After the cache expired (about 6.5 minutes after Create), the profile animation showed the new answer, mid-typing and fully typed.

Elapsed labels: measured from the file timestamps the screenshot tool gives each
capture (frames taken in the same batch share a timestamp, so those labels are
approximate to within 1-2 s). The Create click time is estimated at 0.5 s before
the first post-click frame.

Honest gaps:
- The screenshot taken at the exact instant of the click on the `qa:` link was lost (the tool call dropped its connection); frame 3 is the form that opened from that click, taken a few seconds later.
- The answer is what the model really produced: "Jerry's profile README is at github.com/Bobliuuu/Bobliuuu, and the GitHub Action for README-LM runs from issues opened in that repo." It is not a good answer to the question, only a real one.
- The profile update took about 6.5 minutes to show in the browser, not 20-60 s: the Action finished in about 35 s, but GitHub's raw-file cache served the old image until it expired.
