# Demo 3: a neural network drawn by SVG filters, computed by the viewer's browser

Results returned by the build agent and read back through `decode.py` by the main session (the agent's own numbers were not independently re-run, apart from decoding the sample). `python3 demos/svg/gen.py "<input text>" -o demos/svg/latest.svg` prints `{"input","windows","bytes"}` and writes one SVG. Every multiply-accumulate is an `feConvolveMatrix`, every ReLU or lookup an `feComponentTransfer`, the argmax is `feMorphology`, and the glyph drawing is `feDisplacementMap`. The generator only writes the input as pixels (one-hot codes) around a fixed graph. Measured in headless Chrome (Playwright, installed Google Chrome, software raster, DPR 1, SVG embedded through `<img>` as a README would) by `check.py` (log `check_output.txt`).

## Results

| Stage | Result | Exactness (vs a pure-numpy integer reference, decoded from rendered pixels) |
|---|---|---|
| 1. The MediaWiki toy (2-character context, 12 ReLU units, 5 output bits) | PASS | all 1024 contexts: hidden 12288/12288, logit codes 5120/5120, bits 5120/5120, char id 1024/1024, predicted glyph 1024/1024. The SVG glyph prediction equals the 40 live rows of `trackD/mediawiki/README.mediawiki` and the numpy forward of its model on all 1024 contexts |
| 2. 5-character context, 128 ReLU units, 32 logits, 8-bit QAT (shipped) | PASS | 7 inputs including the empty input and the 97-window maximum: hidden codes, logit codes, argmax ids, every heat pixel, predicted, context and actual glyphs all exact |
| 3. Depth limits | measured | see below |

## Architecture (stage 2)

Rows are windows (24 pixels each); columns are channels. Window j is the 5 characters before position j, left-padded with spaces; the last window predicts the character after the input.

1. Input data (the only thing the Action writes): a PNG `feImage` of one-hot pixels, 5 positions x 32 symbols.
2. Layer 1: 128 `feConvolveMatrix` neurons (161 taps, integer kernels in [-127,127]); the embedding is folded in. The convolve's round and clamp give an 8-bit pre-activation.
3. A 256-entry `feComponentTransfer` table: ReLU and requantize.
4. Layer 2: 32 `feConvolveMatrix` neurons, 8-bit logit codes.
5. Argmax: `feMorphology` dilate, equality LUT, index ramp, `feMorphology` erode (lowest index on ties).
6. Glyph: the predicted id pixel becomes a map pixel and `feDisplacementMap` gathers the glyph from a 16x24 atlas.
7. Heat strip: each hidden column colour-mapped by a LUT, so the viewer sees the 128 activations.

Training (`train.py`): torch, 8-bit activation QAT on `corpus.txt` plus the QA text, lowercased to the 32-symbol vocabulary `a-z space . , - ' ?`. Shipped model: context 5, 128 hidden units: 57.8% train and 57.0% held-out next-character accuracy (mostly in-distribution text; not generalization to arbitrary English). It is a toy: for "the quick brown fox" it predicts `whe awicksteowsearr`.

## Timing and size (render time, `<img>` embed)

| context / hidden | held-out acc. | N=40 | N=100 | N=200 | primitives | KB (N=40) |
|---|---|---|---|---|---|---|
| 4 / 64 | 52.6% | 1.0 s | 1.6 | 3.3 | 465 | 135 |
| 4 / 128 | 56.7% | 2.7 | 3.9 | 6.6 | 597 | 199 |
| **5 / 128 (shipped)** | 57.0% | 2.7 | 4.4 | 7.7 | 597 | 213 |
| 6 / 128 | 56.9% | 3.4 | 5.4 | 9.3 | 597 | 228 |
| 6 / 192 | 60.2% | 5.9 | 9.8 | 14.3 | 661 | 287 |

Render time is linear in windows. The shipped cap is 96 characters (97 windows): 4.1 s, 239 KB. `feTile` re-renders its upstream chain once per repetition; replacing the tiles with per-column `feOffset` copies took 120 windows from 20.6 s to 2.2 s.

## The caveat that decides how it is used

Filters run at device pixels. At a device pixel ratio of 2 (Retina) or when the image is scaled (a narrow phone, a README `width` different from the image's 624 px) the output is **wrong** (heat strip 1708/2560 and predicted glyph 0/20 at DPR 2; 1477/2560 and 0/20 at 0.8x). It is correct only in a Chrome-engine browser at 100% zoom on a 1x display. No workaround was found. So the profile README never embeds this graph directly: the Action renders it in its own headless Chrome (1x), reads the predicted glyphs back with `decode.py`, and shows them in the animation; the live graph is linked for viewers who want to see their own browser compute it.

## Stage 3: depth limits (`stage3.py`)

A random integer MLP of width W with per-neuron convolves: width 32 is fine to depth 3 (2.8 s at depth 4, crash at depth 5); width 64 to depth 3 (0.86 s), crash at depth 4; width 128 takes 11 s at depth 3. Every completed run was exact. Cost jumps by about the layer width per extra depth, then the renderer dies (consistent with `trackD/svgops`). Depth 2 at width 128 is the practical ceiling for one image, so this model spends parameters on width and context, not depth.

## Safety (`check.py safety`)

Input is NFKD-folded, lowercased, reduced to the 32-symbol vocabulary (other characters become a space), whitespace collapsed and capped at 96 characters. The sanitized string is only used as pixel values and glyph indices; it never appears in any text node. 12 hostile inputs passed (`</svg><script>`, `"><image href=x onerror=...>`, `&lt;svg onload=...&gt;`, `{{x}}`, RTL, emoji, NUL, and others). No generated file contains script, onload, onerror, javascript:, foreignObject, `<style`, `<use`, `<a `, or an http(s) URL apart from the xmlns; every href is a `data:image/png;base64,` URI.

## Files

`gen.py` (CLI), `decode.py` (renders in headless Chrome and reads the predicted glyphs back), `svgnn.py` (primitive ops), `build.py` (builders and layouts), `train.py` (QAT), `model.npz`, `model_atlas.npz` (glyph atlas; `gen.py` needs only numpy), `check.py`, `harness.py`, `stage3.py`, `check_output.txt`, `stage3_output.txt`, `latest.svg` (sample for "the quick brown fox").
