# Track C: CSS + variable-font substrate (baseline; CSS is the last resort)

Findings returned by the probe agent (not independently re-run by the main session). Tested on headless Chrome 156 with screenshots decoded by PIL; no JS in any shipped test page (a few debug probes used test-only JS via DevTools). Nothing was committed or published. Code and artifacts are in this directory (`c1/`..`c4/`, `input/`).

## License of the reference repo
`ShubhamPatilsd/gpt2-in-css` has no license (`license: null`; its README says "TBD"). All rights reserved by default: do not copy or adapt its code. Everything here was written from scratch from the blog description (only the repo README was read). Reuse of actual files needs the author's permission.

## Milestones (all PASS)
- C1: 8x8 integer matvec with the matrix as a generated variable font (fontTools HVAR; row = glyph, column = axis; input via `font-variation-settings`; output read as text width). 3 trials, all outputs exact.
- C2: CSS-only math on registered `@property` numbers: ReLU, sigmoid-GELU, exp, pow, sqrt, 4-way softmax with max-subtraction, RMSNorm, RoPE via sin()/cos(), round/mod/clamp, tanh-via-exp. 22 cases, errors <= 0.0025. Gotcha: `+` and `-` inside `calc()` need surrounding spaces.
- C3: 2-layer MLP (4 -> 8 -> ReLU -> 4) chained entirely in CSS via a scroller marker with a named `view-timeline`, `timeline-scope`, and paused keyframes on a registered `<number>`. 3 trials within 1.5 px of numpy.
- Extras: 256-axis fonts work, a 257-axis font is silently rejected; a two-axis HVAR region multiplies axes (gives attention q.k and p*v); typed input path works (below).

## Porting the real QA model (d=128, ff=512, 3 layers, 4 heads)
- Vocab is 85 chars + EOS = 86 rows. 17 fonts, 611,840 weights, 4.74 MB ttf, 0.56 MB woff2 (about 0.7 MB with tokenizer and attention fonts). `dn` (ff=512 inputs) needs two 256-axis fonts added together.
- Precision (numpy simulation): 9/8/6-bit per-row weights with 1/64-unit output quantisation gave greedy output identical to the float reference on 30/30 held-out questions. Not modeled: float32 CSS arithmetic and accumulation over a multi-pass loop (the blog saw RMS error 10.36 in one connected graph versus 0.045-0.074 when each block settles as its own pass).
- Op mapping: tokenizer, matmuls, RMSNorm, RoPE, scores tested; softmax over hundreds of terms and the weighted V sum layout not built; argmax and answer printing untested; KV cache freeze trick not tested; the autoregressive pass loop (SMIL `<set>` counter + container style queries + content-visibility gating) is the largest unproven part.
- Time model from the blog (about 1.07e-4 s per element): about 0.4 + 0.042*T seconds per token. Typical answer N about 170 tokens: about 11 min; N=300: about 34 min; N=410: about 65 min; uncertainty 2-3x either way. Capping context at about 192 gives about 6-10 min.
- Effort estimate: about 3-4 weeks for one engineer from scratch; a toy 1-layer, T<=48 version in about 1 week.
- Chromium-desktop only and a heavy tab; the minimum Chrome version is unverified (none of the Chrome 139 features the blog needs were used).

## Input (typed prompt, no JS): PASS
A `<style contenteditable>` element is typable and Chrome reshapes its text live. A `len` font (every in-vocab char advance 1) gives the character count; a `tok` font (chained GSUB lookups plus a `POSN` HVAR axis) decodes the k-th char to its id (16 ids decoded exactly). No BPE needed for the char vocabulary. Gotchas: some PUA/invisible start-marker characters split the shaping run (U+2060 works); the cmap must cover every typable character.

## Hosting
GitHub Pages (plain HTML + CSS + woff2, `.nojekyll`), linked from the README; the README sanitiser strips `<style>`, fonts and view timelines. A `<meta http-equiv="Content-Security-Policy" content="script-src 'none'">` proves no JS.

## Verdict
Feasible but slow, with one real engineering risk (the autoregressive control loop). Ranked last by project priority: try the other substrates first.
