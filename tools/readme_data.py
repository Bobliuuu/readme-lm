#!/usr/bin/env python3
"""Derive numbers for README.spec.yml from files that already live in this repo.
Writes results/derived/readme_data.json. Nothing here is typed by hand."""
import glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
out = {}

# GPT-2 stored as Markdown shards
shards = sorted(glob.glob(P("gpt2/weights/part-*.md")))
out["gpt2"] = {"shards": len(shards), "mb": round(sum(os.path.getsize(s) for s in shards) / 1e6)}
# QA model weights block
out["qa"] = {"weights_kib": round(os.path.getsize(P("qa/weights.md")) / 1024)}

# QA click graph
qa = json.load(open(P("graph/qa/stats.json")))
out["qa_graph"] = {"questions": qa["questions"], "states_raw": qa["states_raw"],
                   "states_small": qa["states_after_chain_contraction"],
                   "shrink_x": round(qa["states_raw"] / qa["states_after_chain_contraction"], 1)}

# GPT-2 link graphs: chance a continuation is still inside the graph after 8 tokens
g = json.load(open(P("graph/gpt2/stats.json")))
out["gpt2_graphs"] = {"prompts": len(g), "states_max": max(r["states_raw"] for r in g)}
out["coverage"] = sorted(({"prompt": r["prompt"].replace("\n", " ")[:34], "pct": round(100 * r["mass_d8"], 1)} for r in g),
                         key=lambda r: -r["pct"])

# SVG filter ops, checked bit-exact in headless Chrome
ops = json.load(open(P("trackD/svgops/results_all.json")))
out["svgops"] = {"ops": len(ops), "pass": sum(1 for o in ops if o["PASS"]), "mismatches": sum(o.get("mismatches", 0) for o in ops)}
out["svgops_rows"] = [{"op": o["op"][:34], "prims": o["prims"], "kb": round(o["svg_bytes"] / 1024)} for o in ops if o["op"].startswith("linear")]

# MediaWiki net, verified on GitHub's renderer
rep = open(P("trackD/mediawiki/REPORT.md")).read()
m = re.search(r"(\d+)/(\d+) live next-character predictions", rep)
out["mediawiki"] = {"ok": int(m.group(1)), "of": int(m.group(2))}

# executable documents / sizes
out["mdrt"] = {"docs": len(glob.glob(P("mdrt/*.md")))}
out["qa_facts"] = {"files": len(glob.glob(P("qa/*.jsonl")))}
os.makedirs(P("results/derived"), exist_ok=True)
json.dump(out, open(P("results/derived/readme_data.json"), "w"), indent=1)
print(json.dumps(out)[:600])
