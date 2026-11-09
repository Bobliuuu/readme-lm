# Track C: Jekyll/Liquid as the compute substrate

Substrate: Jekyll + Shopify Liquid 4.0.4 (the version GitHub Pages ships), rendering Markdown files with `{% assign %}`, `for`, `if`, `include`, and weights in `_data/*.json|yml`. No JS, no custom plugin, no Action doing arithmetic.

## Tooling actually used (real Jekyll, not a substitute)
- Ruby 3.4.2 (Homebrew). Installed `jekyll 4.4.1` (user gem; uses liquid 4.0.4) and a second sandbox install of `jekyll 3.10.0` (the exact Pages version) with liquid 4.0.4. Both built with `--safe` (Pages forces safe mode). Outputs of L1-L4 are byte-identical across 3.10.0 and 4.4.1; timings differ by about 15 percent (3.10.0 faster). Not tested: the real Pages builder (Ruby version/CPU unknown, github-pages gem not installed).
- Machine: Apple-silicon Mac. A Pages build runner is probably not faster.

## GitHub Pages facts (from docs.github.com / pages.github.com/versions)
- Pages gem: Jekyll 3.10.0, Liquid 4.0.4 (pages.github.com/versions).
- Limits page: published site <= 1 GB; source repo recommended <= 1 GB; **deployments time out after 10 minutes** (confirms the recollection); soft limits 100 GB/month bandwidth, **10 builds/hour** (not applicable if a custom Actions workflow builds+publishes). No per-file limit stated on that page (GitHub's general 100 MB hard file limit in git applies; I did not verify that page).
- Plugins: only the Pages-supported gem whitelist; unsupported plugins fail. Docs example config sets `safe: true`; Pages is documented here only as "some settings can't be changed"; the docs I fetched do not spell out what safe mode restricts. Safe mode disables custom plugins/`_plugins`, not Liquid, and everything below ran with `--safe`. Docs say nothing about build-time limits in the legacy Jekyll builder beyond the 10-minute deployment timeout; I could not confirm whether that limit covers the Jekyll build step itself (assume yes).
- Caveat: Pages-with-Actions ("custom workflow") can use any Jekyll version, but the task excludes Actions we author for compute; the stock Pages build is what counts.

## Milestones (all vs numpy, reproduce with `check.py`)
| | Result | Detail |
|---|---|---|
| L1 4x4 int matvec, `_data/l1.yml`, page `site/l1.md` | **PASS** | exact integer match `[-124,-163,-66,52]` |
| L2 ReLU / exp / softmax / sqrt, `site/l2.md`, `_includes/exp.html` | **PASS** | see precision below |
| L3 token loop, `site/l3.md` | **PASS** | greedy toy RNN-LM generates `[5,3,3,3,3,0]` = numpy; stops on EOS with `break`; KV-cache analogue via `push` |
| L4 2-layer MLP 4-8-3 (floats, JSON), `site/l4.md` | **PASS** | max abs err 3e-16 over 5 inputs |
| L5a throughput | measured | below |
| L5b `_data` with all 4-bit weights loads | **PASS** | 14 JSON files, 1.79 MB total (int values + fp scales), loaded and indexable; trivial-page build 1.1 s, about 48 MB RSS |
| L5c embedding + RMSNorm + qkv0 linear (384x128), token `q`, `site5/l5.md` | **PASS** | max abs err 1.3e-15 vs float64 numpy reference on the same dequantised weights (float32 `qa_infer` would differ ~1e-6); 2.1 s total build |

### L2 precision (float64 in Liquid; grid-scanned vs numpy)
- ReLU: `at_least: 0`, exact.
- exp: range reduction x = k ln2 + r, Taylor, rescale by 2^k via loop. x in [-30,10], max rel err: degree 6 = 1.5e-7, degree 8 = 2.4e-10, degree 10 = 2.6e-13. Softmax of a 6-vector: max abs err 4e-15. (Model weights are 4-bit, so degree 6 is already far below the quantisation noise; cost is ~k+deg assigns.)
- sqrt / rsqrt: Newton y <- y(1.5 - 0.5 v y^2), no division. From a fixed start y=0.5 and v in [0.05,4]: 4 iters 48 percent err, 8 iters 5e-4, 10 iters 2e-13, so a fixed start needs range-normalisation. In L5 I scale v into [0.25,1) by x4 loop steps and start at 1.4: 8 iters converge to double precision for any magnitude (used for the RMSNorm, error 1e-15).
- No `sqrt`, `exp`, `sin`, `cos` filters exist in Liquid. RoPE cos/sin would be a constant table in `_data` (or Taylor like exp). GELU needs erf: the repo's erf approximation uses exp(-x^2), so it reuses the exp include.
- Floats: `times`/`divided_by` with a float literal or float variable stay float64 (Ruby Float). Integer `divided_by` floors, which is useful for group index `j / 32`.

### L3: how a token loop works
- Liquid `assign` inside a `for` is not scoped: state persists across iterations, so a generation loop is just `for step in (0..N)` with state variables, `push` to append (hidden gem: Jekyll/Liquid array filter, works in Pages), and `break` for EOS. No unrolling needed. The hard bound is that `N` is fixed before the build, and that range expressions are evaluated at loop start.
- Recursive `include` works but nests only to **depth about 95-98** (`Liquid::StackLevelError: Nesting too deep`; probe in `depth_probe.rb`). Loops are the right tool.
- No data-dependent loop other than `break`; everything is a bounded `for`.

## L5 throughput (MAC = one `assign acc = x[j] | times: w | plus: acc`)
- 128x128 matvec (16,384 MAC), 100 repeats: **Jekyll 4.4.1: 0.20 s per matvec = ~82K MAC/s; Jekyll 3.10.0: 0.14 s = ~115K MAC/s**. Loop unrolling 4x and `for w in row` variants gave no real gain (~100K MAC/s is the ceiling; Liquid is a tree-walking interpreter allocating per `assign`).
- Real op chain: qkv0 linear 49,152 MAC took about 0.7 s of a 1.8 s build, consistent.
- Cost per position (one char through 3 layers): linears 589,824 MAC; attention 6*128*T (T=100: 77K); GELU (512x3, each ~35 ops via exp include) ~54K; softmax exps ~50K; RMSNorm/RoPE/residuals ~30K; head 11K only when sampling. Total about **0.8M ops, so ~8 s per position at 100K/s (~10 s at 82K/s)**.
- A 100-char answer: ~35 prompt positions + 100 generated = ~135 positions = **~17-22 minutes in one build**. That is 1.7-2.2x over the 10-minute deployment timeout, on a fast laptop, before any slowdown of the Pages runner. About 60-75 positions fit in 10 min, i.e. roughly 30 answer chars after a 35-char prompt in a single build (optimistic).
- Memory: whole weight set in `_data` ~48 MB RSS for Jekyll; no problem. Data size 1.79 MB JSON (<< 1 GB).

## The INPUT question
Pages rebuilds only on commit/push, so the prompt must be committed content (front matter or a `_data` file). Options:
1. Human edits a prompt file and commits (zero Actions). Works, pure Pages, but not interactive.
2. Issue-driven: a stock Action (e.g. on `issues: opened`) writes the issue text (sanitised to vocab, lowercased, 140 chars as in `qa_infer.answer`) into `_data/prompt.json` and commits. That Action does no arithmetic; Pages then does all of it. Issue: the Action is authored by us (the task says no authored Actions); and committing with `GITHUB_TOKEN` does not trigger a Pages build by itself in some configurations (needs a PAT/`workflow_dispatch`-style trigger or the Pages build via the token-triggered "pages-build-deployment" workflow, which does run for pushes by the token in practice but verify).
3. Chunked generation to beat the timeout: each build computes K positions (say 25, ~4 min) and emits `state.json` (KV cache, 3 layers x 2 x T x 128 floats, ~0.8 MB at T=100, via `jsonify`). A copy-only Action downloads the published `state.json` and commits it as `_data/state.json` plus the next token; the next build resumes. ~6-7 builds for a 100-char answer. Constraints: the soft 10 builds/hour limit, deploy latency of minutes per hop, and the fact that the "answer" only exists across many commits. The step "copy the previous output into the repo" is data movement, not arithmetic, but it is still an authored Action.
4. Build times are quadratic if state is not persisted (recompute the prefix every build), so persistence (3) is required for long answers.

## Verdict (honest)
- Feasible in principle: a transformer's arithmetic *can* be expressed in Jekyll/Liquid. Integer and float matvec, ReLU, a Taylor exp/softmax, Newton rsqrt, an autoregressive loop with a KV-cache-style array, and the first op chain of the real 600K 4-bit model (embedding, RMSNorm, linear) all render and match numpy to 1e-15; 1.8 MB of weights loads from `_data` fine, under `--safe`, on the Pages Jekyll/Liquid versions.
- Not practical for the real model on stock Pages: ~100K MAC/s gives ~8 s per character position and ~17-22 min per 100-char answer, exceeding the 10-minute deployment limit even on a fast laptop. Whole-answer-in-one-build is out; only about 30 answer chars fit, and only if the runner is as fast as this Mac. Multi-build chunking works but needs an authored copy-only Action and many commits, and is subject to the 10 builds/hour soft limit.
- Not done (honest gaps): full single-layer attention + MLP + head in Liquid (the op chain stopped at the first linear); the real Pages runner speed; exact Pages file-size limits; enforcing of safe mode specifics. Remaining ops are straightforward by the same patterns, so the expected result of finishing is a correct but ~8 s/char model.
- A smaller model (say d=32, 2 layers, ~50K params) would be ~12x cheaper per position (< 1 s) and could answer within one build.

Files (all in this directory): `site/` (L1-L4 pages and includes), `site5/` (L5 chain + 1.8 MB `_data`), `bench/`, `bench_tpl/` (throughput), `gen_small.py`, `gen_model_data.py`, `check.py`, `depth_probe.rb`, `l2_analysis.txt`, refs `ref_small.json`, `ref_l5.json`. Run: `cd site && jekyll build --safe` (add `GEM_HOME=/tmp/gh310` for the 3.10.0 install), then `python3 check.py`. Nothing committed or published; no IPs/tokens used.
