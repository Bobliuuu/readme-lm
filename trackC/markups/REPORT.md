# Render-time compute in non-Markdown markups GitHub renders (Track C / markups)

Date 2026-11-08. All local emulation; nothing was published to github.com.
Scripts and raw outputs are in this directory: a.adoc, e.adoc, f.adoc, w.rb, w2.rb, w3.rb, gh.rb, gh.out.txt, o.org, r.rst.

## What GitHub uses (confirmed from github/markup source via WebFetch + installed github-markup 6.1.1)
| Ext | Renderer | Notes |
|---|---|---|
| .adoc/.asc/.asciidoc | Asciidoctor 2.0.26 (gem) | github-markup passes attributes only (showtitle, env=github, ...); `Asciidoctor.convert` default safe mode is SECURE. (the asciidoctor.rb file 404'd on master; I read the same block in the installed gem's markups.rb) |
| .mediawiki/.wiki | WikiCloth 0.8.3 | `WikiCloth::WikiCloth.new(:data => content).to_html(:noedit => true)` - no link/template handler is supplied |
| .rst | docutils via commands/rest2html (python3) | settings: file_insertion_enabled False, raw_enabled True (!), math_output latex, report_level 5, syntax_highlight none |
| .org | org-ruby | |
| .textile / .creole / .rdoc / .pod | RedCloth / creole / rdoc / pod2html | not run locally (RedCloth/creole not installed); no compute features known |
| sanitization | downstream GitHub HTML pipeline (not in the gem) | |

Env: gems installed fine with `--user-install` (wikicloth needed `brew install libidn` for idn-ruby, plus small Ruby 3.4 shims in shim.rb for twitter-text/File.exists?/frozen string; the shims only fix emulation, not behaviour). docutils installed via pip (v0.23 here; GitHub's version unknown).

## Milestones
Inputs/outputs are exact (HTML trimmed to the relevant text).

| Milestone | AsciiDoc (secure) | MediaWiki (WikiCloth) | reST (docutils) | Org / Textile / Pod / RDoc / Creole |
|---|---|---|---|---|
| M1 const compute (7+5) | PASS only via counters: `{counter2:s:7}{counter2:s}x5` then `M1 s={s}` -> `M1 s=12` (unary increments; `{a}+{b}` stays literal `7+5`; ifeval arithmetic fails) | PASS: `{{#expr: 7+5}}` -> `<p>12.0</p>` | FAIL: `\|a\|+\|b\|` is text substitution; `.. math:: 7+5` -> `7 + 5` (formatting only) | FAIL (org `#+BEGIN_SRC emacs-lisp (+ 7 5)` is dropped, not executed; `#+TBLFM` ignored; `{{{a}}}` macro not expanded) |
| M2 conditional | PARTIAL: `ifeval::[{a} > {b}]` with literal attrs -> true branch rendered (constant comparison only). ifeval takes `lhs op rhs`, no arithmetic (`{a} + {b} == 12` -> not rendered). ifeval cannot see counter values (counters expand at convert time, ifeval at read time): `{s} == 12` with s=12 -> not rendered | PASS: `{{#if: x\|YES\|NO}}`->YES; `{{#ifeq: 12\|12\|EQ\|NE}}`->EQ; `{{#switch: b\|a=A\|b=B\|#default=D}}`->B; `{{#ifexpr: 7+5 = 12 \| T \| F}}`->T | FAIL (no conditionals; `only` is Sphinx) | FAIL |
| M3 bounded loop/accumulation | PARTIAL: `{counter:n}` x3 -> `1 2 3`; counter2 accumulates (above). Repetition is author-written, no loop construct; include directive is disabled so no recursion | FAIL on GitHub: no templates exist (default include_resource returns ""), so no macro to iterate. Even with an emulated handler, recursion never terminates (see M5) | FAIL | FAIL |
| M4 4x4 matvec / dot | FAIL (no multiplication; only +1 increments) | PASS (unrolled): `{{#expr: 1*5 + 2*6 + 3*7 + 4*8}}` -> 70.0; rows `{{#expr: 1*1+2*2+3*3+4*4}},...` -> `30.0,6.0,12.0,10.0`; ReLU `{{#ifexpr: {{#expr: 1*2-3*4}} > 0 \| {{#expr: 1*2-3*4}} \| 0}}` -> 0 (nested #expr works) | FAIL | FAIL |
| M5 recursion / Turing | FAIL: `include::a.adoc[]` in secure mode becomes a link `<a href="a.adoc" class="bare include">`; no loops | FAIL on GitHub (no template source). In emulation: direct self-recursion is detected ("Template loop detected"); mutual recursion A<->B with `#ifexpr` guard overflowed the stack even for depth 1, because args are evaluated eagerly (no lazy branch), so no terminating recursion. Not Turing complete here | FAIL | FAIL |

## MediaWiki details worth knowing
- Supported in wikicloth default_functions: #if, #switch (string compare), #expr, #ifexpr, #ifeq, #iferror, #len, #sub, #pad, lc/uc, #tag, plural. `#vardefine/#var` give empty output. Lua extension not configured.
- `#expr` is a small parser: + - * / mod, comparisons, parentheses. `^`, round, abs, max, sqrt, e-notation all error ("Unknown token ^10"). Results are Ruby floats: `12.0`. `#ifeq` compares strings: `{{#ifeq:{{#expr:7+5}}|12|..}}` -> NE (12.0 != 12); `|12.0|` -> EQ. `#switch` has the same float-vs-int problem.
- Output is HTML `<p>12.0</p>`; GitHub's downstream sanitizer not tested.
- Inline `<template 1="X">{{{1}}}</template>` is just substitution; there is no way to define a reusable template inside the page.

## Not verifiable without publishing to github.com
- That GitHub's production versions/configs equal the open-source ones used here (docutils version, Asciidoctor version, WikiCloth patches).
- Downstream sanitizer / render timeouts / size cap (I recall a ~512 KB display limit but did not verify).
- Whether GitHub supplies any template resolver for wikis in repo files (source shows none).
- Whether ifeval/counter behaviour differs under GitHub's pipeline.

## Verdict
Render-time compute in a README is real only in two places: MediaWiki `#expr/#ifexpr` (a genuine, if tiny, arithmetic evaluator with comparison) and Asciidoctor counters (unary increment only, invisible to conditionals). reST, Org, Textile, Pod, RDoc, Creole: nothing.

But a README is a static function of its file: there is no viewer input, so any "computation" evaluates a constant expression the author wrote. The output equals what the author could have typed directly; the render only demonstrates that the evaluator ran. For a transformer it could at most show a fixed forward pass on a fixed prompt (precomputed-equivalent).

MediaWiki stunt, quantified:
- Straight-line only: no templates, variables, loops or recursion on GitHub. A network must be fully unrolled into one nested `#expr` text tree.
- No variables means no sharing: a hidden unit used by n downstream units must be textually duplicated, so size grows ~n^(depth-1) for fan-out n. A 4-4-4 MLP with ReLU: layer 1 ~4 expressions of ~25 chars each, but layer 2 inlines each of those 4 times: low thousands of characters, fine. A 2-layer width-64 MLP inlines 64 copies of 64 neurons per output (~4k MACs per output, ~6 chars each -> ~25 KB per output). Depth 3+ at width 16 or more quickly hits MBs. Practical ceiling: a few hundred to a few thousand MACs at depth <=2 within a ~512 KB file (unverified cap), nowhere near a real LM layer (millions of MACs).
- Float64 only, `^`/exp/sqrt/tanh unavailable, so no softmax/GELU except via piecewise #ifexpr/#switch approximations; #switch needs 12.0-vs-12 normalisation.
- WikiCloth is a char-by-char Ruby parser; deep nesting will be slow (timeouts unverified).
- Output is one-shot, non-interactive. Honest conclusion: a stunt demonstrating a toy 4x4 matvec + ReLU at render time is plausible (M1, M2, M4 pass in emulation); anything resembling transformer inference, or any dependence on viewer input, is not.

## Correction (2026-11-08)

WikiCloth 0.8.3's `{{#ifexpr: ...}}` always takes the true branch (Ruby treats 0 as truthy), so the earlier M2 pass was only a true-case check. Use `{{#ifeq:{{#expr:a>b}}|1|then|else}}` for conditionals. Found while building `trackD/mediawiki/`, which was then verified on GitHub's live renderer.
