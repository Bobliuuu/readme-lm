# Tweet thread

Attach the listed file(s) under each tweet (MP4 on X; the GIF is for the README and other sites).

**Note on the draft.** The original draft was: "I put GPT-2 completely inside markdown, a transformer in a README, and an MLP inside an SVG." "GPT-2 completely inside markdown" overclaims: the weights and the program are Markdown files, but the engine that runs them is Python on a GitHub Action. Honest wording: "I put GPT-2's weights and program into Markdown files that a small Python engine runs on a GitHub Action, a tiny transformer inside a profile README, and a small MLP inside an SVG and a MediaWiki page."

1. I put GPT-2 into Markdown, a transformer into a README, and a small MLP into an SVG. Honest version of what that means, in a thread. (Note: GPT-2's arithmetic is done by a minimal Python engine on a GitHub Action, not by Markdown itself.)
   Attach: `docs/video/explainer.mp4` (or `docs/clips/2-three-versions.mp4`)

2. The question: a README can't run code, so who does the arithmetic? Three answers, from a tiny engine on an Action down to no Action at all.
   Attach: `docs/gifs/arch-three-versions.mp4`

3. Version A: real GPT-2 small (124M, int8) stored as about 339 Markdown shard files. A minimal Python engine on a GitHub Action reads them and runs the forward pass. The Action only does I/O.
   Attach: `docs/gifs/arch-fast-version.mp4`

4. A 600K-parameter character-level transformer QA model (4-bit weights) is small enough that its weights fit inside a profile README.
   Attach: `docs/gifs/arch-qa-transformer.mp4`

5. Version B: a small MLP written as MediaWiki formulas. GitHub's renderer computes the numbers. It is an MLP, not a transformer, and no Action is needed to compute it.
   Attach: `docs/gifs/mediawiki-render.mp4` (mechanism: `docs/gifs/arch-mediawiki-net.mp4`)

6. Version C: a small MLP written as SVG filters. Your browser computes it. Also an MLP, not a transformer; the transformer ops were built and checked op by op, but chaining them does not render in Chrome.
   Attach: `docs/gifs/svg-net-reference.mp4`

7. Why 8 bits matters for SVG filters: retraining at 8-bit activations makes the net match the reference.
   Attach: `docs/gifs/precision.mp4`

8. Where it runs and how fast: the Action engine answers in about 0.2 s; pure-markup routes take minutes or more.
   Attach: `docs/gifs/time-per-answer.mp4`

9. Code and details: https://github.com/Bobliuuu/readme-lm
   Attach: `docs/gifs/poster.mp4`
