"""Held-out-ish check: 2nd phrasing of each menu item (not in the first-phrasing eval set; they were in training with typo aug)."""
import sys, os, json, numpy as np
sys.path.insert(0, "/Users/jhome/readme-lm/trackD/qat8"); sys.path.insert(0, "/Users/jhome/readme-lm")
from multiprocessing import Pool
import build_qa, qa_infer, q8_ref
def init():
    global A, B; os.chdir("/Users/jhome/readme-lm"); A = qa_infer.load("/Users/jhome/Bobliuuu/README.md"); B = q8_ref.Q8Ref("/Users/jhome/readme-lm/trackD/qat8/q8_params.npz")
def run(q): return qa_infer.answer(A, q), B.generate(q)
if __name__ == "__main__":
    os.chdir("/Users/jhome/readme-lm"); items, _ = build_qa.load(); qs = [o["qs"][1] for o in items if len(o["qs"]) > 1][:350]
    with Pool(10, initializer=init) as p: res = p.map(run, qs, chunksize=4)
    ex = [a == b for a, b in res]; print("alt-phrasing exact", sum(ex), "/", len(ex), np.mean(ex))
