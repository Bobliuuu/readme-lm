# Track D2: SVG filter op library for the QA transformer

Results returned by the build agent (not independently re-run by the main session). Headless Chrome driven by Playwright; generated SVGs contain no JavaScript. Scratch SVGs (`work/`, about 100 MB) were deleted; code, logs and `results_*.json` are here. The 8-bit model and integer reference this targets are in `../qat8/` (`q8_spec.md`, `q8_ref.py`).

## Verdict

Every op is bit-exact in headless Chrome, but the full model is **not feasible as one README `<img>` with this design**. A deep chain of per-neuron convolves has exponential render cost in Chrome: the full token step (4,827 primitives, 3.17 MB at T=8) did not render within 240 s. The largest verified single-graph chain is embedding -> RMSNorm -> QKV -> RoPE -> full causal attention (T=8/16/64 PASS in-pipeline). Everything after the attention output (`L0.ao`) is unverified in Chrome; the output projection after attention crashed the renderer.

## Op ladder (26/26 op tests PASS, bit-exact against an integer numpy mirror; `results_all.json`)

| Op | Result | Primitives / SVG size / render time |
|---|---|---|
| O1 linear 128->128, 128->384, 128->512, 512->128, real weights | PASS | 132 to 519 prims, 85 to 335 KB, about 0.1 to 0.3 s each |
| O2 LUTs (GELU, exp, reciprocal, rsqrt, square) | PASS | 2 prims, 6 KB, 0.04 s |
| O3 RMSNorm (reduction, rsqrt LUT, scale) | PASS | 11 prims |
| O4 residual add (3 scale settings) | PASS | 3 prims |
| O5 embedding lookup 86x128; argmax over 86 logits (lowest index on ties) | PASS | 132 prims, 110 KB; 14 prims |
| O6 RoPE (per-position cos/sin rasters) | PASS | 16 prims |
| O7 causal attention T=8/16/32/64 (4 heads x 32) | PASS, 0 mismatches | 67 to 68 prims; 27/39/62/103 KB; 0.16/0.61/4.2/20 s (about T^2.4) |
| O8 glyph atlas via `feDisplacementMap` (token id pixel -> visible character) | PASS | 7 prims |
| row gathers for attention | PASS, T=8 to 64 | about 30 prims, constant in T |

`<img>` embedding was tested for the 128->128 linear only; it matched direct rendering.

## Spec conformance

The generator used the final `q8_params.npz` / `q8_spec.md`, including the fine max-subtracted score stage. The integer mirror in `pipeline.py` is the agent's own (8-bit-integer kernels instead of the spec's per-group multipliers, norm gains folded into the next linear): logits within about 1.5 levels (mean) of `q8_ref.py`, next-token argmax equal on 7 of 8 test prompts at T=16. The claim "the predicted id equals the spec reference" was never tested through Chrome, because the full graph never rendered.

## Findings that drive the design

- Skia quantizes a convolve kernel to 8 bits over [kmin, kmax]; a kernel is exact only if (max - min) divides 255. A guard tap on a zero pixel forces range 255, so integer kernels up to +-127 give bit-exact sums. With a +0.25/d nudge to avoid ties, out = floor(S/d + zp + 0.5).
- The filter region must contain the whole kernel window; partial windows give garbage.
- `feComposite` arithmetic needs alpha_out = k1+k2+k3+k4 >= 1, which forbids plain `a-b` (worked around with negate-then-add).
- `feDisplacementMap` gives exact integer gathers only for even coarse steps; `feMorphology` is an exact sliding max/min up to radius 127 and replaced the argmax and max tournaments (argmax 84 -> 14 prims).
- **Skia does not memoise shared filter nodes across consumers with different rects**: cost multiplies per fan-out. Stacked 128-wide convolve layers cost 0.18 s at depth 1, 0.23 s at depth 2, 7 s at depth 3, and crash at depth 4. This is why the full chain fails.
- Filters run at device pixels: at DPR 2 the feImage and convolve results change, so the graph is valid only at 1x (Retina or zoomed viewers would be wrong); no workaround found.

## Weights

Kernel text is about 3.2 B/weight (1.93 MB for 601K weights; 1.78 MB at wmax=63). A `feImage` PNG raster is smaller (about 0.72 MB) but a convolve kernel cannot be loaded from it; it would need a product-based layer (weight raster x broadcast input, then grouped convolve sum), whose per-term 8-bit rounding gave mean error 0.65 to 1.3 levels, max 4 to 6 (numpy only, not built in SVG).

## What would unblock it (untested)

A product-based dense layer (about 30 primitives per layer, each node used once) with weight rasters and a 2D transposing gather via displacement maps. Estimated at roughly another day of work. Even then: no input is possible in an `<img>` (the context must be static data), Chrome-only, DPR=1 only, attention render time about T^2.4 (extrapolated to minutes at T=192), and GitHub's behaviour for a multi-megabyte filter SVG beyond the 8.2 MB padded test in `../githubtest/` was not checked with a real model graph.
