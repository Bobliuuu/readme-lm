#!/usr/bin/env python3
"""Regenerate graph/qa/profile-menu.md: a standalone ~140-question click page (local #qNnM anchors).
Selection: ask.PRESETS, all 'curated' items, the 70 shortest 'profile' items, up to 3 shortest per other repo, capped by size.
Usage: [QA_CKPT=qa_final3.pt] python3 graph/make_profile_menu.py [nodes_per_question=260]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compile_qa as C, build_qa, ask
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 260; CAP = 60_000; NPROF, NREPO = 70, 3
items, _ = build_qa.load(); ST = json.load(open(f"{ROOT}/graph/qa/stats.json"))
byrepo = {}
for o in items:
    q = build_qa.norm_q(o["qs"][0])
    if q: byrepo.setdefault(o["repo"], []).append((len(build_qa.clean_a(o["a"])), q))
sel = []; seen = set()
def add(q):
    if q and q not in seen: seen.add(q); sel.append(q)
for q in ask.PRESETS: add(build_qa.norm_q(q))
for _, q in sorted(byrepo.get("curated", [])): add(q)
for _, q in sorted(byrepo.get("profile", []))[:NPROF]: add(q)
for r in sorted(byrepo, key=str.lower):
    if r not in ("curated", "profile"):
        for _, q in sorted(byrepo[r])[:NREPO]: add(q)
head = f"""# QA click-through menu (moved from the profile README)

This page used to sit at the bottom of the profile README. It is the 4-bit QA transformer's behavior precomputed into links; nothing runs when you click. The full {ST['questions']}-question version is in [`README.md`](README.md). (Answers here are from the model as it was when this menu was compiled and are refreshed when the model is retrained.)

## Ask the model with no Action (pure Markdown)

This is the 4-bit transformer's behavior **precomputed into links**: pick a question, then click through the answer. Each link spans the run of characters the model is confident about, and a state with several links is a place where the model had real alternatives (probabilities shown). Nothing runs when you click; GitHub just scrolls to the next precomputed state. More questions: [the full graph](https://github.com/Bobliuuu/readme-lm/blob/main/graph/qa/README.md).

"""
trees, size, kept = [], len(head.encode()), []
for q in sel:
    md, _ = C.render_tree(len(kept), q, C.build(q, N)); cost = len(md.encode()) + len(q) + 30
    if size + cost > CAP: break
    kept.append(q); trees.append(md); size += cost
menu = "".join(f"- [{C.esc(q)}](#q{i}n0)\n" for i, q in enumerate(kept))
open(f"{ROOT}/graph/qa/profile-menu.md", "w").write(head + menu + "\n" + "\n".join(trees))
print("menu questions", len(kept), "of", len(sel), "bytes", size)
