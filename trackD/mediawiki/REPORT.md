# MediaWiki neural net computed by the renderer (Track D / mediawiki)

Nothing was committed, pushed or published. Local emulation only: WikiCloth 0.8.3 and github-markup 6.1.1 (setup reused from trackC/markups: `shim.rb`).

## Files
- `gen_mediawiki.py H D NROWS` : trains the integer model (torch QAT, 6 seeds, best kept; run as `python3 gen_mediawiki.py 12 6 40`), writes `model.npz` and `README.mediawiki`.
- `README.mediawiki` : the page (310,872 bytes). `render.rb` : renders with WikiCloth, or with `github-markup` (`... github`). `check.py` : renders both ways, parses the HTML tables, compares with an independent python-int forward pass from `model.npz`.

## The model
Two input characters (32-symbol vocab) -> embeddings (6 dims each, 12 inputs) -> 12 hidden ReLU units -> 5 output bit-logits (binary-coded output, bit = z > 0) -> 5-bit string decoded to a character by `#switch`. All weights are integers in [-7,7] (biases wider), so float64 `#expr` arithmetic is exact. Trained on 194 characters of the project's own README text; quantised train exact-match 63.6% (honest: a toy; 12 hidden units cannot memorise 100+ contexts).

## What the renderer computes vs what is precomputed
Computed live by the renderer: every multiply-accumulate of both layers (`#expr`), the ReLU (`#ifeq` on a `#expr` comparison), the output sign tests, the bit-string -> character `#switch`. Page has 40 rows (first 40 distinct 2-char contexts of the text, no cherry-picking), each a self-contained ~7.5 KB expression with the 12 hidden units recomputed inside each of the 5 logits; plus section 2 showing hidden vector, 5 logits and bits for one input, live next to numpy values.
Precomputed/in the text: trained weights (literals), which contexts each row uses, and the embedding gather (the two embedding vectors of the row's input are written as literals; no variables/loops to look them up). A README has no viewer input, so every row is a constant expression; it demonstrates a genuine forward pass evaluated by the renderer, not interactivity.

## Results (check.py)
- WikiCloth: section 1 40/40 live predictions == numpy; section 2 hidden 12/12 and logits 5/5 exact, max error 0.0, 0 bit mismatches. Same through `GitHub::Markup.render("README.mediawiki", ...)`: identical.
- Overall exact-match rate 100%, max error 0.
- Render time: 1.2 s total per render (ruby start + parse), output HTML only ~8.5 KB (expressions vanish into results).

## Quirks found in WikiCloth 0.8.3 (design consequences)
1. `#ifexpr` is broken: Ruby `if val` treats 0 as true, so `{{#ifexpr: 3>5|T|F}}` -> T (the trackC M2 PASS on `#ifexpr` was only a true case). It only yields the else branch on a parse error. So the page never uses `#ifexpr`; comparisons use `{{#ifeq:{{#expr:a>b}}|1|then|else}}` (comparison returns Integer 1/0, so no 12.0-vs-12 problem).
2. The expression grammar has no unary minus and comparisons are only legal at the top level (not inside parentheses: `(3>2)` errors). Hence all substituted intermediate values must be non-negative: each pre-activation is written as P > N and P-(N) with the positive-product terms P and negative terms N split at generation time (note `P-(N)` needs the parentheses; N is a sum). ReLU outputs are therefore >= 0 and safe to substitute. Logits are only displayed at top level (negative is fine there).
3. `#ifeq` treats hex-looking strings numerically; irrelevant here (compared "1"). Results of `#expr` arithmetic are floats ("12.0") but nothing downstream compares them as strings.

## Size growth (measured, 40 rows)
| H | D | acc | bytes | per row |
|---|---|-----|-------|---------|
| 8 | 4 | 45% | 173 KB | 4.3 KB |
| 12 | 6 | 64% | 311 KB | 7.8 KB |
| 16 | 6 | 69% | 434 KB | 10.8 KB |
Per cell: size ~ NB * H * (2*(2D)*5 + overhead): linear in H, D and output bits; each hidden unit is duplicated per output bit and twice inside the ReLU (P and N both appear twice) because there are no variables. Binary-coded output (5 bits) instead of 32 logits + argmax is what makes it fit: a 32-way argmax needs ~V^2 duplicated logit copies. A third layer multiplies cost by the fan-in again (~H x); a depth-3 net of width 12 is ~100 KB per bit per cell, i.e. a handful of cells at most.
Greedy chain (output of step t fed into step t+1) was not built: the previous char expression would be duplicated once per hidden unit x2 (ReLU) x5 bits ~ 100+ times per level, so depth 2 is already ~1 MB at this width.

## Honest limits / what could break on github.com (unverified)
- Production WikiCloth version/patches/config: unknown. If github.com uses a build where `#expr`/`#ifeq`/`#switch` differ, or a template resolver, results change; the quirks above are version-0.8.3-specific.
- GitHub's downstream HTML sanitizer, render timeout, and the ~500 KB display limit are not verified. File is 310 KB; the rendered HTML is tiny, so only the source-size limit and render time (1.2 s locally, a Ruby char-by-char parser) matter. If the blob view shows "too big", cut rows via `NROWS` (e.g. 20 -> ~160 KB).
- If github serves a raw-file fallback (no render), the page shows the literal markup; the prose says so.
- Only exact integer arithmetic is guaranteed; float errors absent by construction (max |z| ~ 1e4).

## Verify after publishing
GitHub renders server-side, so `curl -s https://github.com/<owner>/<repo>/blob/<branch>/README.mediawiki` (or the repo main page for README) contains the computed table. Grep for the live column: no `{{#` in the table (only inside the `<code>` prose), no "Expression error", and compare the "live" and "numpy ref" columns: `python3 check.py` logic applies to the saved HTML (point `out_wikicloth.html` at the curl output and reuse the table parser). Mismatch of live vs ref columns or leftover `{{` text means the production renderer differs.

## Live check on github.com (2026-11-08)

Fetched GitHub's own server-side rendering of the pushed page (`gh api -H 'Accept: application/vnd.github.html+json' repos/Bobliuuu/readme-lm/readme/trackD/mediawiki`, saved as `out_live.html`) and ran the same comparison as `check.py` (`check_live.py`):

- section 1: 40/40 live next-character predictions equal the python-int reference
- section 2 (input `we`): max |hidden error| 0.0, max |logit error| 0.0, 0 bit mismatches, hidden 12/12 and logits 5/5 exact
- no `Expression error`, no unevaluated `{{#` outside code spans

So GitHub's production renderer evaluates every `#expr` in the page and produces the same numbers as the numpy reference. Still unverified: the interactive web blob view in a browser (the API render comes from the same renderer), and how GitHub's limits behave if the page grows past about 500 KB.
