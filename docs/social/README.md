# Social media assets: THE RULE

**EVERY diagram and figure, in the README and in tweet media, must be an ANIMATED GIF (and also an MP4 for X). NO STATIC PNGs as deliverables. EVER.**

- Static diagrams are only sources (`docs/img/*.svg|png`). The deliverables are `docs/gifs/<name>.gif` + `docs/gifs/<name>.mp4`.
- Regenerate with `python3 docs/tools/make_gifs.py [name ...]` (Playwright + ffmpeg; GIF <= ~1.5 MB, width <= 860, 12 fps, silent H.264 MP4, crf 27, yuv420p, faststart).
- A new figure is not done until it has a GIF and an MP4 and the README references the GIF.
- `docs/social/` holds the thread text (`TWEET.md`) only. Do not put PNGs here.
- Public content shows current correct results only.
