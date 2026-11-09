# Demo 2: MediaWiki page that takes input; the renderer does the arithmetic

Nothing committed or published. Local emulation: WikiCloth 0.8.3 and github-markup (setup from trackC/markups, `render.rb`).

## Usage
`python3 demos/mediawiki/gen.py "<input text>" -o demos/mediawiki/README.mediawiki` prints `{"input": sanitized, "windows": n, "bytes": size}`.
Weights: `model.npz` (copied from trackD/mediawiki; 2-char context, 6-dim embeddings, 12 ReLU hidden units, 5 output bits, 32 symbols `a-z space . , - ' ?`).
`check.py` renders with WikiCloth and GitHub::Markup and compares against an independent python-int forward pass. `check_live.py out_live.html` does the same on GitHub's own render:
`gh api -H 'Accept: application/vnd.github.html+json' repos/Bobliuuu/readme-lm/readme/demos/mediawiki > out_live.html`.

## Who computes what
- Generator (text writing only, no network evaluation): sanitizes the input, writes the trained integer weights, the embedding-vector numbers of the input characters (as literals in each expression) and the sanitized characters as numeric entities.
- GitHub's renderer: every multiply-accumulate of both layers (`#expr`), every ReLU (`#ifeq` over an `#expr` comparison), every sign test (`#expr` comparison) and the 5-bit decode (`#switch`). The page contains no precomputed prediction; the "actual next" column is merely the next input character.
- The last window has the full worked example: 12 hidden units, 5 logits, 5 bits, decoded character, all live.

## Sanitizing and injection safety
Lowercase, whitespace collapsed, every character outside the 32-symbol vocabulary dropped (so no `{ } [ ] < > | = # : & ~ * ! 0-9` etc.), leading/trailing space stripped. The raw input never reaches the page. Remaining characters are written as numeric entities (`&#39;` etc., space as the visible `&#9251;`), so even `''` or `--` cannot form markup. The only parser functions in the output are `#expr`, `#ifeq`, `#switch` and the only tags `<code>`/`<b>`, written by the generator. Empty/unusable input falls back to `hello` (stated on the page); one char gets a leading space. Input is passed as an argv element (no shell interpolation by gen.py).
Tested hostile inputs: `{{#expr:1}}` (becomes "expr"), `[[x]] [http://e.x y]`, `<script>alert(1)</script>`, `|}` `== x ==` `{| class=a` table fragments, `{{#switch:...}} ''it's'' '''b''' ~~~~ __TOC__ &#60;`. check.py asserts per page: whitelisted functions/tags only, no `[[`, no `http`, no `<script`, size <= 420000 bytes, and the window text echoed back by the render equals the sanitized input.

## Size and limits
About 7.3 KB per window plus about 25 KB fixed (header, worked example). Cap: 420,000 bytes, i.e. input truncated to the first N characters (N-1 windows) found by binary search, typically 54 windows / 55 characters (the exact number varies a little with the embedding values of the characters). The page states when it truncated. Render time locally 0.3 to 1.2 s; the rendered HTML is small.
Quirks of the renderer (from trackD): no `#ifexpr` (buggy), no unary minus, comparisons only at top level, hence the P/N split with non-negative intermediates. Entities inside `<code>` are double-escaped by WikiCloth, so input characters are shown outside `<code>`. The model is a toy (about 64% exact next-char on its training text); the point is that the renderer computes, not that predictions are good. Unverified: GitHub's production limits beyond what check_live.py shows.

## Results (check.py, 11 inputs x 2 renderers)
short, long-truncated (54 windows, 419,334 B), all-vocabulary, five hostile inputs, single char, no usable chars, mixed case/whitespace: 886/886 live cells exact (100%) in both WikiCloth and GitHub::Markup; all hygiene checks pass. Sample: `README.mediawiki` for "the quick brown fox" (18 windows, 156 KB), 41/41 cells exact through `check_live.py` on the local render.
