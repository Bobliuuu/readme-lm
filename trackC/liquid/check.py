import json, numpy as np, re
def nums(p): return [float(v) for v in re.sub(r"</?p>", "", open(p).read()).split()]
r = json.load(open("ref_small.json"))
print("L1", "PASS" if nums("site/_site/l1.txt") == r["l1"] else "FAIL")
print("L3", "PASS" if [int(v) for v in re.sub(r"</?p>|GEN", "", open("site/_site/l3.txt").read().split("CACHE")[0]).split()] == r["l3"] else "FAIL")
g = np.array(nums("site/_site/l4.txt")).reshape(-1, 3); print("L4", "PASS" if abs(g - np.array(r["l4"])).max() < 1e-9 else "FAIL", abs(g - np.array(r["l4"])).max())
q = np.array(nums("site5/_site/l5.txt")); ref = np.array(json.load(open("ref_l5.json"))["qkv"]); print("L5-chain", "PASS" if abs(q - ref).max() < 1e-9 else "FAIL", abs(q - ref).max())
