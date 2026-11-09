# Track C / SVG filters: can a renderer (no JS) run the README-LM transformer?

Renderer: headless Chrome (macOS, software raster, `--headless=new --disable-gpu`), SVG opened directly AND via `<img>` in an HTML page.
Every test below gave identical results direct vs `<img>`. Chrome with GPU enabled gave identical results on the chain test.
rsvg-convert also tried: it is NOT compatible (8-bit chain quantizes differently; the 4-8-4 MLP fails: kernel semantics differ).
Code: h.py (harness), s1.py s1b.py s1c.py s2.py s2b.py s3.py s4.py s5a.py s5scale.py sim8.py sim8b.py. Artifact: mlp_4_8_4.svg (24 MLPs, 300 KB).

## PASS/FAIL

| # | Milestone | Result | Numbers |
|---|-----------|--------|---------|
| S1 | a*b with feComposite arithmetic (k1) | PASS | 2704 (a,b) pairs, sRGB: 100% == round(a*b/255) |
| S1 | a+b (k2,k3) | PASS | 100% == round((a+b)/2) up to .5 ties (max err 1 on ties) |
| S1 | color-interpolation-filters | MUST use sRGB | linearRGB: product max err 14 levels, only 23% exact; sum max err 60 |
| S2 | 4x4 int matvec via feColorMatrix (W as matrix, x in RGBA pixel incl. alpha) | PASS | 12/12 exact, W in [-3,3], x in [-7,7], offset-128 encoding |
| S2 | same via feConvolveMatrix (order 4x1, W as kernel) | PASS | 30/30 matvecs exact (120 outputs). NOTE Chrome applies the kernel flipped (spec 180 deg rotation); bias attr works as +bias*255 |
| S3 | ReLU + arbitrary LUT via feComponentTransfer (type=discrete or table, 256 entries) | PASS | relu, GELU-table, random permutation: 256/256 exact. Table entries must be (v+0.5)/255 (v/255 truncates and is off by 1 on ~60% of entries) |
| S4 | 4-8-4 int MLP (convolve per neuron, offset+merge to assemble, LUT ReLU, bias) | PASS | 24 random nets: 24/24 exact logits vs numpy, argmax 24/24 (direct and `<img>`). Weights W1 in [-2,2], W2 in [-1,1] |
| S5 | Real model (600K, 3x d128, 4-bit) | NOT BUILT; feasibility + simulation, verdict below | |

## Precision findings (S1/S5)
- Every primitive output is an 8-bit premultiplied RGBA value, clamped to [0,1]. Chain x*0.1 then *10 via two feComposite: values quantized to 0..25 then 10x (visible loss). linearRGB adds sRGB<->linear conversions in 8 bits: unusable.
- Exception: adjacent feColorMatrix primitives are fused in float by Skia (x*0.1 then *10 returned exact x). Do not rely on it (rsvg does not do it).
- No higher precision path found (GPU on = same). Wider numbers can only be emulated by splitting across channels (hi/lo) with LUT carries: costs primitives, not tried.
- feConvolveMatrix with integer weights is exact. With fractional kernels (needed for scaling) Chrome shows +0.7..+1 level mean bias and max error 1 level (n<=128), about 2 levels (n=128 x 384 outputs, 51% exact), and 4 levels at n=256 (observed offsets, cause not identified; likely Skia internal rounding). Treat linear layers as +-1.5 level noise.
- HARD LIMIT: feConvolveMatrix kernel <= 256 entries (order 256x1 ok, 257x1 and 16x17 return transparent/0). 512-wide FFN down-projection must be split in 2 convolves + feComposite add (extra rounding).
- 4-bit weights fit trivially (weights are kernel numbers; integer kernel + divisor attribute gives scale without loss).

## S5 op-by-op

| Op | Filter expression | Status |
|----|-------------------|--------|
| embedding lookup | stored context -> one-hot rows (static rects) x emb matrix as 90->128 convolves, or direct emb rows as constant feFlood/feImage data | OK (it is data, not arithmetic) |
| RMSNorm | square = feComposite k1 (x*x); mean = convolve of 128 ones; rsqrt = LUT; x*r*g = two products (8-bit each) | OK, 3-4 roundings |
| qkv / o / up / down linear | one feConvolveMatrix per output neuron (order <=256), rows = tokens so all tokens in parallel; dn needs 2 convolves + add | OK. Exact for integer, +-1.5 level noise with scale |
| RoPE | x1*c - x2*s with c,s a constant per-(pos,dim) data image, feComposite products + add | OK |
| attention scores q.k | per relative offset d: feOffset dy=d on K, product with Q, per-head convolve sum (32 wide); ~T offsets x 4 heads x 3 layers | OK but graph grows with T (T~160: several thousand primitives) |
| softmax | exp via LUT; needs per-row max subtraction (tournament via offset + sub + LUT-ReLU + add, ~8 rounds); sum via convolve; 1/sum LUT; product | OK but the per-row max is required (see sim) |
| P.V | products + add over offsets (T adds, each rounds to 8 bits) | weakest point: accumulation of T 8-bit terms; better to sum with convolve after transposing layout |
| GELU | feComponentTransfer LUT (exact 256-entry) | OK, exact |
| tied head (90 logits) | 90 convolves of order 128 | OK |
| argmax | tournament as above; indicator via LUT; display char by multiplying per-glyph feImage bitmaps by indicator and summing | plausible, not built |

### Precision budget (sim8.py: numpy model with every boundary quantized to N bits, static per-tensor scales calibrated on 8 questions, 40 test questions, greedy decode, vs fp-activation 4-bit-weight reference)
| activation bits | exact string match | mean char match |
|---|---|---|
| 8 | 87.5% | 0.908 |
| 8 + uniform +-1.5 level noise on linear outputs (Chrome convolve) | 80% | 0.844 |
| 10 | 95% | 0.972 |
| 12 | 95% | 0.981 |
| 16 | 97.5% | 0.993 |
- Per-tensor ablation at 8 bits: only unnormalized softmax exp (no max subtraction) is catastrophic (e range up to 7e5); with fixed offset it is also catastrophic; per-row max subtraction required. All other tensors individually near 100%.
- This is PTQ only (no 8-bit activation QAT) and ideal LUT-composed ops (division, rsqrt not separately re-rounded). With QAT/PTQ tuning the 8-bit result should improve; with real per-primitive rounding it will degrade somewhat. So: 8-bit is borderline-viable, not obviously broken.

### Graph size / runtime estimate
- ~8K convolve neurons (qkv 384 + o 128 + up 512 + dn 128x2 per layer, x3, + 90 head), each also needs offset + merge node => ~16K primitives for linears; attention/softmax per layer adds a few thousand (T offsets); total ~20-30K primitives.
- Text size: ~3-4 MB (measured: one 128->384 layer with 16 tokens = 0.67 MB with decimal kernels; integer kernels + divisor ~2.5 B/weight, 600K weights = 1.5 MB + ~1.2 MB structure).
- Measured: 128->512 layer, T=160 rows, 2 MB SVG rendered correctly in ~3 s in headless Chrome (but screenshots intermittently stalled ~45 s, unexplained; real viewer behaviour on 25K-primitive graph unmeasured).
- Everything recomputes from scratch per page view (no KV state): prefill over the stored context (<=~160 tokens) per viewer per render.

## Input question / README viability
- An `<img>`-loaded SVG has no input. Human-in-the-loop design: SVG stores the context as static data (rects/consts) and renders computed next token; README holds one static prefilled-issue link per vocabulary char (91 links, a "keyboard") or one generic link where the viewer types the char (the link in markdown cannot depend on the image output); a GitHub Action parses the issue title and appends the char to the SVG's context string and commits: zero arithmetic. This works logically. Costs: one Action run + commit + CDN/camo cache refresh (minutes) per token => a 150-char answer = 150 round trips: not usable as chat; fine as a demo of "the renderer did the arithmetic for one token". Viewer picks the displayed char, so it is trusted-human, not enforced autoregression (anyone can append anything; could validate nothing without arithmetic). Option B (cheating-ish): Action runs headless Chrome and reads the pixel; arithmetic then happens in Chrome in CI, not in the viewer.
- README rendering (reasoned, NOT verified, nothing published): relative `<img src=model.svg>` in a repo README is served as image/svg+xml with a locked-down CSP/sandbox; `<img>` rendering in the browser still runs the full filter pipeline incl. feConvolveMatrix/feComponentTransfer/feComposite (confirmed locally via `<img>` in HTML) and CSS/SMIL animation but no scripts and no external refs (feImage must use data: URIs; our design uses none). Externally hosted images go through camo, which proxies image/svg+xml but has a size cap (believed ~5 MB) and fetch timeouts: a 3-4 MB SVG is near the limit and unverified. SVG sizes of several MB may also slow page load. Unverified: GitHub's exact size cap, whether GitHub (or its CDN) rewrites/sanitizes SVG (I believe repo SVGs are not sanitized), and Firefox/Safari behaviour.
- Cross-browser risk: rounding, convolve limits and premultiply handling are implementation-specific. Chrome passes; rsvg demonstrably differs; Firefox/Safari untested. Model would need to tolerate +-1-2 level noise.

## Verdict
- Primitives work: exact integer matvec, exact LUT nonlinearities, product/sum, and a tiny integer MLP (S1-S4 PASS, direct and via `<img>`), Chrome only.
- The real 600K transformer is expressible op-by-op (nothing fundamentally missing; softmax needs max tournament, attention needs T relative-offset copies) but a faithful build is big: ~20-30K primitives, ~3-4 MB SVG, hard 256-entry kernel cap, 8-bit activations everywhere. Idealized 8-bit simulation keeps 87.5% exact strings (80% with Chrome convolve noise), so it is plausible with 8-bit QAT retraining, and likely ~95% at 10-12 bits (needs channel-splitting tricks).
- Not done: building any piece of the real model in SVG, argmax-in-filter + glyph display, runtime of a full graph, cross-browser, GitHub camo/size behaviour. Honest call: PASS for "a renderer can do neural arithmetic"; "the full README-LM model inside a README SVG" is feasible but unproven, risky on size (camo cap), render time, and non-Chrome browsers; interactive loop is one token per Action round trip.
