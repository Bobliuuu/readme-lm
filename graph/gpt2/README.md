# GPT-2 in pure Markdown

GPT-2 small (124M), compiled into link graphs. **No Action, no JavaScript, no server: you click.** Each prompt below opens a page whose states are the real GPT-2's top next-token predictions. Pick a token to extend the text; the next state shows the text so far.

Honest limits: probabilities are GPT-2's at temperature 0.6 (sharpened, as in typical sampling); only the most probable branches (top-3, probability >= 8%) are compiled, up to a state budget, so you can only reach continuations GPT-2 rated likely; paths end at a depth limit. Compiled from the int8 weights stored in [`../../gpt2/`](../../gpt2/).

| Prompt | States | P(text stays in graph for 8 tokens) | for 12 tokens |
|---|---|---|---|
| [The capital of France is](p01.md#n0) | 1713 (from 2000) | 11.5% | 0.0% |
| [Question: What is the capital of France? Answer:](p02.md#n0) | 1371 (from 2000) | 54.0% | 28.7% |
| [Question: Who wrote Hamlet? Answer:](p03.md#n0) | 1636 (from 2000) | 8.7% | 0.0% |
| [To be, or not to be,](p04.md#n0) | 1634 (from 2000) | 2.1% | 0.0% |
| [Roses are red, violets are](p05.md#n0) | 1472 (from 2000) | 0.2% | 0.1% |
| [1, 2, 3, 4, 5,](p06.md#n0) | 1497 (from 2000) | 70.6% | 64.2% |
| [The Eiffel Tower is located in](p07.md#n0) | 1538 (from 2000) | 16.7% | 5.7% |
| [Water boils at 100 degrees](p08.md#n0) | 1650 (from 2000) | 5.0% | 0.0% |
| [The meaning of life is](p09.md#n0) | 1679 (from 2000) | 5.7% | 0.0% |
| [Once upon a time](p10.md#n0) | 1694 (from 2000) | 0.7% | 0.0% |
| [GitHub is](p11.md#n0) | 1644 (from 2000) | 8.0% | 0.0% |
| [Markdown is](p12.md#n0) | 1599 (from 2000) | 1.3% | 0.0% |
