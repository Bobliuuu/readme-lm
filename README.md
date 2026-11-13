<div align="center">

<img src="assets/hero.svg" alt="README-LM" width="100%">

# 🧠 README-LM

**A real neural language model compiled into a README, and an honest map of who does the arithmetic**

![versions: 3](https://img.shields.io/badge/versions-3-58a6ff) ![GPT-2: 124M params, int8](https://img.shields.io/badge/GPT--2-124M%20params%2C%20int8-d95926) ![QA model: 600K params, 4-bit](https://img.shields.io/badge/QA%20model-600K%20params%2C%204--bit-d95926) ![MediaWiki: computed by GitHub](https://img.shields.io/badge/MediaWiki-computed%20by%20GitHub-1a7f37) ![SVG filters ops: 26/26 bit-exact](https://img.shields.io/badge/SVG%20filters%20ops-26%2F26%20bit--exact-1a7f37) ![python: 3.x + numpy](https://img.shields.io/badge/python-3.x%20%2B%20numpy-3776ab?logo=python&logoColor=white)

[🎬 Video](#video) · [✨ The idea](#idea) · [📊 Numbers](#results) · [🧩 How it works](#how-it-works) · [🕹️ Try it](#try) · [⚠️ Honest limits](#caveats) · [📚 Full README](#full) · [🗺️ Repo map](#repo-map)

</div>

---

**Explainer video (with voice-over and captions): the puzzle, the three versions, what is measured, and the limits**

[![Explainer video preview (click for the mp4)](video/preview.gif)](video/readme_lm_explainer.mp4)

<a id="video"></a>
A README is a static document: it cannot run code. So "a language model in a README" really asks one question: **who does the arithmetic?**
This repo builds three answers and is explicit about what actually runs. **Version A** stores a real GPT-2 small (124M parameters, int8, 339 Markdown shards, about 150 MB on disk)
and a 600K-parameter question-answering transformer (weights file 411 KiB) as Markdown, and a tiny engine written in Markdown runs them on a GitHub Action.
**Version B** is a small MLP written as MediaWiki formulas that GitHub's own renderer evaluates. **Version C** is a small MLP drawn as SVG filters that your browser evaluates.
B and C are toys (not transformers), and nothing we tried runs a GPT-2-sized model inside the renderer itself.

<a id="idea"></a>

## ✨ The idea in one minute

<div align="center"><img src="docs/gifs/arch-three-versions.gif" alt="The three versions: A normal (minimal Action), B pure MediaWiki, C pure SVG" width="860"></div>

| | 🟠 **A · Normal** | 🟢 **B · Pure MediaWiki** | 🟢 **C · Pure SVG** |
|---|---|---|---|
| **What it is** | real **GPT-2 small** (124M, int8) and a **600K QA model** that knows my repos | a **small MLP written as MediaWiki formulas** (not a transformer) | a **small MLP drawn as SVG filters** (not a transformer) |
| **Who computes** | a ~150-line engine inside a Markdown document, run by one line of Python on a GitHub Action | **GitHub's own renderer** evaluates every `{{#expr}}` | **your browser's filter engine** |
| **Needs an Action?** | yes, a minimal one | no | no |
| **Honest scope** | real models; a minute or a few per reply | a toy (2-character context, 12 units) | a toy (7-character context, 224 units); Chrome at 100% zoom on a 1x display |

<a id="results"></a>

## 📊 Numbers from the repo

Every number below is read from files in this repo by `tools/readme_data.py`.

| GPT-2 shards | SVG ops bit-exact | QA graph shrink | QA weights file |
|:---:|:---:|:---:|:---:|
| **339** | **26** | **51x** | **411 KiB** |

**SVG filter linear layers, bit-exact in headless Chrome (trackD/svgops/results_all.json)**

| Op | Filter primitives | SVG size (KiB) |
| :--- | ---: | ---: |
| linear 128->128 (o0) | 132 | 83 |
| linear 128->384 (qkv0) | 390 | 240 |
| linear 128->512 (u0) | 519 | 327 |
| linear 512->128 (dn0) | 394 | 300 |
| linear 128->128 (o1) via <img> | 132 | 84 |

<sub>Source: `results/derived/readme_data.json`</sub>

<p align="center"><img src="assets/coverage.svg" alt="GPT-2 link graphs: chance a continuation is still in the graph after 8 tokens" width="100%"></p>

<p align="center"><img src="assets/qa_graph.svg" alt="QA click graph: states before and after merging confident runs" width="100%"></p>

The MediaWiki page was checked against GitHub's production renderer: 40 of 40 live next-character predictions equal the integer reference ([trackD/mediawiki/REPORT.md](trackD/mediawiki/REPORT.md)).
Per-op SVG results: [trackD/svgops/results_all.json](trackD/svgops/results_all.json). Everything else (precision study, substrates tried, character model, timings) is in the [full README](docs/README_FULL.md).

<a id="how-it-works"></a>

## 🧩 How it works

The weights are Markdown (GPT-2 as 339 base64 shards; the QA model as one block). The engine is Markdown too: one document carries the interpreter source, the forward-pass program and the run request. All an Action does is extract and run it.

<p align="center"><img src="assets/flow.svg" alt="Version A: issue to Action to profile animation" width="100%"></p>

<p align="center"><img src="docs/gifs/arch-fast-version.gif" alt="The fast version, end to end" width="85%"></p>

<p align="center"><img src="docs/img/gpt2-generation.gif" alt="Real GPT-2 generating text one token at a time with its top-3 probabilities" width="70%"></p>

<details>
<summary>Version B: pure MediaWiki</summary>

GitHub renders `.mediawiki` files. The small feed-forward network (not a transformer) is written as MediaWiki formulas, so GitHub's renderer does every multiply-accumulate when the page loads. It is a toy: 2-character context, 12 hidden units.
<div align="center"><img src="docs/gifs/mediawiki-render.gif" alt="The MediaWiki page as GitHub renders it" width="620"></div>

</details>

<details>
<summary>Version C: pure SVG</summary>

A 7-character-context, 224-unit MLP written as SVG filter primitives (`feConvolveMatrix`, `feComponentTransfer`, `feMorphology`, `feDisplacementMap`). It is not a transformer, and it only renders correctly at 1x device pixels.
<div align="center"><img src="docs/gifs/svg-net-reference.gif" alt="Reference render of the SVG network" width="620"></div>

</details>

<a id="try"></a>

## 🕹️ Try it

Open an issue in this repo and replace `example text` in the title: the prefixes are `gpt2:`, `qa:`, `mediawiki:` and `svg:`. An Action runs the engine and rewrites the animation at the top of the profile.
<div align="center"><img src="docs/gifs/issue-flow.gif" alt="Clicking a link opens an issue with the title prefilled; the Action replies with the answer" width="720"></div>

Deploy it on your own profile with `scripts/deploy.sh --dry-run` then `scripts/deploy.sh` (see [DEPLOY.md](DEPLOY.md)).
Run the executable document yourself or GPT-2 from the shards: `python3 gpt2/run.py "The meaning of life is" 60`.

<a id="full"></a>

## 📚 Full original README

The complete previous README (all substrates tried, the precision study, the character model, reproduction commands, credits) is kept as [docs/README_FULL.md](docs/README_FULL.md).

<a id="caveats"></a>

## ⚠️ Honest caveats

- The two shipped versions are not the same thing: the fast version computes outside GitHub's renderer, on a GitHub-hosted runner; the pure-markup version computes inside it, but only as a toy.
- No substrate we tried runs a GPT-2-sized model inside a README. GitHub's renderer has no input-dependent arithmetic, so anything computed there is a fixed function of the file.
- The QA model is mostly a lookup table in weights: near-exact phrasings work, unusual ones can fail. Its training facts were written by agents and not independently verified.
- GPT-2 is a base language model, not a chat assistant; it continues text and can be wrong, repetitive or odd.
- The SVG-filter version chains only as a two-layer MLP; the individual transformer ops are exact but the full chain does not render in Chrome.
- Probe results come from local emulation of GitHub's renderers unless a report says it was checked on github.com.

<a id="repo-map"></a>

## 🗺️ Repo map

| Path | What it is |
|---|---|
| `qa_model.py` | the QA transformer: model, data, training, packing |
| `gpt2/` | GPT-2 Markdown shards, manifest, run.py |
| `mdrt/` | the executable Markdown documents GPT2.md and QA.md |
| `graph/` | compiled click-through graphs and their compilers |
| `trackD/` | probe reports and builds: GitHub renderer, other substrates, MediaWiki and SVG filters |
| `docs/` | figures, GIFs and README_FULL.md (the complete earlier README) |
| `tools/readme_data.py` | derives README and video numbers from repo files |
| `README.spec.yml` | spec for video-readme-maker |
| `video/` | explainer mp4, srt, poster, preview.gif (generated) |

---

<div align="center"><sub>Generated with video-readme-maker from README.spec.yml and files in this repo.</sub></div>
