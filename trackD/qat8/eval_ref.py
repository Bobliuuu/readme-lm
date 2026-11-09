"""eval_ref.py PARAMS.npz [N] : greedy answers of q8_ref vs fp 4-bit model answers (teacher_answers.json)."""
import sys, json, difflib, numpy as np
from multiprocessing import Pool
import q8_ref
T = json.load(open("/Users/jhome/readme-lm/trackD/qat8/teacher_answers.json"))
import os
def init(path):
    global M; M = q8_ref.Q8Ref(path, noise=float(os.environ.get('NOISE', 0)), seed=os.getpid())
def run(q):
    a = M.generate(q); return a, M.maxpos
if __name__ == "__main__":
    path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else len(T); items = T[:n] if n < 100 else T
    if n < len(T) and n >= 100: items = T[::len(T) // n][:n]
    with Pool(10, initializer=init, initargs=(path,)) as p: res = p.map(run, [t["q"] for t in items], chunksize=4)
    ex = [r[0] == t["a"] for r, t in zip(res, items)]
    pos = [np.mean([x == y for x, y in zip(r[0], t["a"])] + [0] * abs(len(r[0]) - len(t["a"]))) if False else sum(x == y for x, y in zip(r[0], t["a"])) / max(len(r[0]), len(t["a"]), 1) for r, t in zip(res, items)]
    rat = [difflib.SequenceMatcher(None, r[0], t["a"]).ratio() for r, t in zip(res, items)]
    bad = [i for i, e in enumerate(ex) if not e]
    print(f"exact {sum(ex)}/{len(ex)} = {np.mean(ex):.4f}; mean char-pos-acc (non-exact) {np.mean([pos[i] for i in bad]) if bad else 1:.4f}; difflib ratio (non-exact) {np.mean([rat[i] for i in bad]) if bad else 1:.4f}; max pos {max(r[1] for r in res)}")
    json.dump([{"q": t["q"], "ref": t["a"], "q8": r[0]} for r, t in zip(res, items)], open(path.replace(".npz", "_answers%s.json" % os.environ.get("NOISE", "")), "w"))
