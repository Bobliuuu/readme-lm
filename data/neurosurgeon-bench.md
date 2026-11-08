# Neurosurgeon-Bench

Can an LLM **write** and **edit** neural-network weights directly, with no gradients and no training?
Inspired by "Weight-Space Programming" (Writing Weights Index) and by neuron-level steering of open-weight models.

## Tracks
| Track | Task | Metric |
|---|---|---|
| **W** — Writing Weights | LLM hand-writes all weights of a tiny MLP (8 hidden) for CartPole, MountainCar, Catch | `Index = mean 100·q/(1+q)`, `q = max(0, score−random)/(ref−random)`; matching ref = 50, random = 0 |
| **S** — Neuron Surgery | LLM picks `(layer, neuron, dose)` edits to push an open-weight model toward a persona trait (`evil`, `sycophancy`, `hallucination`) | judged trait shift, coherence, vs. unedited and random-neuron controls |

Track S edit: at the input of `mlp.down_proj`, `act[N] += dose · median_resid_norm / ‖W_down[:,N]‖` on every token. Known published edits can be replayed with `--edit L,N,dose` to replicate; `--agent MODEL` lets a model choose edits from a contrastive-probe candidate list.

Track S is a safety/interpretability measurement: it quantifies how easily models can locate and exploit trait neurons. Do not deploy edited models.

## Quickstart
```bash
pip install -e .            # add .[steer] for Track S
python -m nsbench.baselines # regenerate configs/baselines.json (random floor, CEM ceiling; MountainCar ref is hand-built)
export OPENROUTER_API_KEY=...
nsbench write --models configs/models.txt --rounds 1     # one-shot; --rounds 3 adds score feedback
nsbench steer --target Qwen/Qwen3-8B --trait evil --agent anthropic/claude-sonnet-4.5
```
Results append to `results/*.jsonl`.

## Notes
- Reference is CEM (not PPO) so the repo has no heavy deps; numbers are not comparable to the original PPO-based index.
- Token usage is recorded per model for cost reporting.

MIT licensed.
