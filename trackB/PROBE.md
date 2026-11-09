# Track B probe: does GitHub's Markdown renderer have an input-dependent primitive?

Method: `gh api -X POST markdown -f mode=gfm -f text=...` and `markdown/raw`. Raw outputs in `raw/*.html`.
Caveat: the API returns sanitized HTML only. Client-side JS (MathJax, mermaid, details toggling, hovercards) is not exercised; those items are marked UNVERIFIED.

## Evidence (short)
- style: `<style>p{color:red}</style><p style="color:red">hi</p>` -> `&lt;style&gt;p{color:red}&lt;/style&gt;<p>hi</p>`. The tag is escaped to text and `style=` is stripped. Only GitHub-added `style="max-width: 100%;"` on imgs and `display:inline-block|block` on math-renderer survive.
- task lists: `<input type="checkbox" id="" disabled="" class="task-list-item-checkbox" aria-label="Incomplete task">`. Disabled in rendered files. A raw `<input type=checkbox>` or `<input type=radio>` outside a list item is removed.
- details: `<details open=""><summary>S</summary>body</details>` kept. `name=g` becomes `name="user-content-g"`, so exclusive-group accordions are possible. Open state is not exposed to any CSS or other element, because there is no CSS.
- Stripped, leaving text only: `label`, `form`, `button`, `select`, `option`, `meta`, `base`, `dialog`, `progress`, `meter`, `map`/`area`, `class`, `hidden`, `contenteditable`, `onerror`, `javascript:` hrefs.
- iframe and script: escaped to text (`&lt;iframe ...&gt;`).
- svg with set/animate: whole element removed (`<p></p>`). `<math>` is reduced to the text `x`.
- Kept: `<img srcset>` (the srcset attribute is dropped from the img), `<video controls>`, `<audio>`, `<kbd>`, `<sub>`, `<ruby>`, `<table>`, `<img usemap>`, `<div tabindex=0>`.
- `data:` images: src dropped. SVG images via URL are proxied through camo and rendered as `<img>`.
- picture: `<themed-picture><picture><source media="(prefers-color-scheme: dark)" srcset=camo...><img src=camo...></picture></themed-picture>`. `media="(max-width:600px)"` is also kept. Source URLs are all camo-rewritten.
- `#gh-dark-mode-only` / `#gh-light-mode-only`: fragment is preserved in the camo URL. Display is chosen by GitHub's site CSS (theme).
- math: `<math-renderer class="js-inline-math" style="display: inline-block" data-run-id=...>$\def\a{1}\a+x$</math-renderer>`. Raw TeX is passed through untouched, including `$$..$$` inline and `` ```math `` fences (emitted as `js-display-math`). `$x$` inside backticks stays code. The API does no TeX processing. `\numexpr1+1` is passed through as text (the `$` delimiters are kept), so it is not computed server-side.
- ids: `<a name>` becomes `user-content-foo`, `id=bar` becomes `user-content-bar`, and `name=` on details also gets the prefix. Heading `<h1>Head One</h1>` gets no id in the API output (GitHub's page adds the anchor client-side). Links `href="#head-one"` and `href="#foo"` are unchanged and are resolved by GitHub JS, which maps them to `user-content-*`.
- fences: mermaid, geojson and stl come back as ordinary `<div class="highlight highlight-source-mermaid">` blocks. The API only syntax-highlights them, and mermaid, geojson (json) and stl (stl) get no special elements. Client JS renders them on github.com and the fence text is static.
- mentions, issue refs, emoji, footnotes: expanded server-side from the text. `#1` and footnote links are static anchors.

## Primitive table
| Primitive | Status | Input-dependent? |
|---|---|---|
| `<style>`, `style=` | stripped | no |
| task-list / input checkbox | kept, disabled (README) | no |
| details/summary | kept; `open` and `name` kept | user toggles it; effect is local to the details box, no other element can read it |
| label/for, form, button, select, radio | stripped | no |
| iframe, script | escaped | no |
| svg (animate/set), math/MathML | stripped | no |
| `<img srcset>` | stripped | no |
| picture `<source media>` | kept (themed-picture) | viewer theme/viewport selects one precomputed image |
| `#gh-dark/light-mode-only` | kept | theme selects one precomputed image |
| `$..$` math | `<math-renderer>` + raw TeX, rendered by client MathJax | UNVERIFIED client-side. TeX is fixed by the file bytes, with no input source. |
| mermaid/geojson/stl | static text in highlight div | UNVERIFIED client-side. Text is fixed. |
| `<a name>`/ids/anchors | kept with `user-content-` prefix | URL fragment only scrolls. `:target` is unusable with no CSS. |
| camo images, SVG-as-img | kept | `<img>` is non-interactive; SMIL `begin=click` cannot fire, and CSS media in SVG depends only on theme/viewport |

## Other viewer-controlled inputs
URL fragment (scroll or highlight only; GitHub highlights `#L10` in file views, which is a view feature, not README content), theme, viewport width, locale/timezone (GitHub `<relative-time>` only for its own elements), logged-in state, `<details>` toggle state, text selection or find-in-page. None can feed a value into the content.

## GitHub-native stateful features outside the README (no Actions)
- Issue/PR task-list checkbox: a click makes GitHub edit the body server-side. Displayed progress ("x of y tasks") is a GitHub-computed popcount of checked boxes. Reactions give counters. Each is a state bit or counter set by a human click, and any displayed function of the state is GitHub's fixed one (popcount or sum), not user-defined. A transformer needs a user-chosen nonlinear function (matmul, softmax and so on). Those are not available. Only a human (or a script using the API/Actions) could take a popcount and produce the next input, and a human doing the arithmetic is not autonomous. Autonomous execution requires a trigger (Actions, webhook, app, bot), which is excluded.
- Also not usable: README rendered in repo view cannot embed live issue state without a server-side image/badge service, which is excluded. Even an issue-state badge hosted on GitHub would need an external server.

## Verdict
1. The README is rendered as sanitized HTML from the file bytes alone. Its only viewer-dependent choices are theme (picture/fragment images) and viewport (`<source media>`), and each selects among precomputed variants.
2. There is no CSS, no forms, no label/for, no SVG animation, no script, and no `:target` (nothing to style). A click or input can change what is shown only by navigating or scrolling to a precomputed location, or by toggling a details box that nothing else can observe.
3. Pure function claim: rendering = f(file bytes, theme, viewport) plus presentation-only toggles (details, scroll). Verified on the HTML, UNVERIFIED for client-side MathJax/mermaid. MathJax macros (`\def`) could compute from the fixed file bytes only, so they give a constant. There is no input source, and `\numexpr` is not MathJax.
4. Smallest experiment that would settle the open client-side question: a README with `$\def\a{1}\a$` and details toggles; it would need a live github.com page. The expected result is that it shows no dependence on any input.
5. Conclusion: a GPT-2 or tiny-transformer forward pass cannot run in a README. There is no input-dependent arithmetic primitive. Autonomous generation, where the output token is computed from the previous output, is impossible without a trigger outside the file.
6. The best achievable is a static lookup table, or precomputed fan-out pages with links (each token a link to the next precomputed README or anchor). That is a human-navigated finite-state browser, not computation.
7. Issue task-list popcount and reactions are human-set state with GitHub-fixed aggregation, so they are no autonomous compute either.


## Addendum: stock MathJax capability test (not GitHub's configuration)

Run locally with `mathjax-full` 3.2.2 in node, default settings, TeX packages base + newcommand + configmacros. The live github.com result is in the next section and supersedes the stock-MathJax conclusions below.

| TeX feature | Result |
|---|---|
| `\def`, `\newcommand` | works (`\def\a{7}\a` renders 7) |
| `\def` with delimited arguments (pattern matching) | works and computes: `\def\f#1+#2={#2#1}\f 3+4=` gives 43; `\def\u#1\|{#1#1#1}\u 1\|` gives 111 |
| `\ifx`, `\the`, `\numexpr`, `\count`, `\csname`, `\gdef` | undefined control sequence |
| macro-substitution budget | default `maxMacros` = 1000: about 111 expansions pass, about 1,110 fail with "maximum macro substitution count exceeded" |

Consequence: macro rewriting and delimiter pattern matching give a tiny lookup/rewrite engine, but with no arithmetic primitives, no conditionals, and roughly 1,000 substitutions per expression, and with no viewer input (the expression is fixed by the file). At most it can evaluate a very small fixed function of the document's own text. A transformer forward pass (hundreds of thousands to 124M multiply-accumulates per token) does not fit.


## Live result on github.com (secret gist with 16 probe lines, page text reported by the user)

GitHub's math renderer is MathJax with a base-TeX-only configuration. On the live page:

| Probe | Rendered as |
|---|---|
| `\def\a{7}\a` | the literal tokens `\def \a 7 \a` (not executed) |
| `\newcommand{\sq}[1]{#1+#1}\sq{9}` and the delimited `\def` macros | error "You can't use 'macro parameter character #' in math mode", raw TeX shown |
| `\ifx`, `\the\numexpr`, `\count`, `\csname`, `\require`, every expansion-budget macro chain | raw tokens, not executed |
| `\href{...}{...}` | raw TeX (stripped) |
| `\frac{1}{2}+\sqrt{9}` | renders normally |

Verdict: GitHub's MathJax can typeset a fixed formula and nothing else. It cannot define macros, so there is no macro-rewriting engine, no arithmetic and no conditionals, and it takes no viewer input. A GitHub-rendered README cannot compute a transformer forward pass, or any function, at view time. The earlier stock-MathJax result (macros and delimiter pattern matching work, about 1,000 substitutions per expression) does not apply to GitHub, which does not load those packages.
