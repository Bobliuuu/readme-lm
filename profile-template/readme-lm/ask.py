#!/usr/bin/env python3
"""ask.py README.md OUT.svg SEED [QUESTION] : answer a question about Jerry's GitHub using the model stored IN the README.
Prints the answer; writes a typing-animation SVG."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_infer, gen

PRESETS = ["what is qwentpu", "which repo runs doom inside qwen", "what is the ten percent challenge", "what language is allgit written in",
           "what is qweninqwen", "where does jerry study", "what is bobliu-os", "what is README-LM"]

if __name__ == "__main__":
    readme, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
    q = sys.argv[4].strip() if len(sys.argv) > 4 and sys.argv[4].strip() else PRESETS[seed % len(PRESETS)]
    q = "".join(c for c in q if 32 <= ord(c) < 127)[:140]
    m = qa_infer.load(readme); ans = qa_infer.answer(m, q)
    meta = f"{m['params']//1000}K-param transformer · 4-bit · lives in this README"
    open(out, "w").write(gen.svg(q, ans, "", meta, label="qa"))
    print(ans)
