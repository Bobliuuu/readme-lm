"""Pool agent-written Q&A into training sequences (question lowercased; loss on answer only)."""
import json, glob, random, re
from qa_model import VOC
REFUSE = ["I can only answer questions about Jerry Zhu's GitHub: his repos, projects and stack. Try asking about qwentpu, allgit or kvm.",
          "That is outside what I know. I only answer questions about Jerry Zhu's GitHub repos and projects, like qwen-os or ten-percent-challenge.",
          "I only know about Jerry Zhu's GitHub. Ask me about one of his repos, for example qweninqwen, bobliu-os or cakebench."]
PRE = ["", "", "", "hey ", "so ", "quick question: ", "tell me ", "can you tell me ", "i wonder ", "pls "]

def norm_q(q):
    q = q.lower(); q = "".join(c for c in q if c in VOC); return re.sub(r"\s+", " ", q).strip()[:140]

MAXA = 150
def clean_a(a):
    a = "".join(c for c in a if c in VOC and c != "\n")
    if len(a) <= MAXA: return a
    parts = re.split(r"(?<=[.!?]) ", a); out = ""
    for p in parts:
        if len(out) + len(p) + (1 if out else 0) > MAXA: break
        out = (out + " " + p).strip()
    return out if len(out) >= 30 else ""

def typo(q, rng):
    if len(q) < 8 or rng.random() > 0.25: return q
    i = rng.randrange(1, len(q) - 1); m = rng.random()
    return q[:i] + q[i + 1:] if m < 0.4 else (q[:i] + q[i + 1] + q[i] + q[i + 2:] if i + 2 < len(q) else q) if m < 0.8 else q[:i] + q[i] + q[i:]

def load():
    items = []
    for f in sorted(glob.glob("qa/g*.jsonl")):
        for l in open(f):
            try: o = json.loads(l)
            except Exception: continue
            if o.get("qs") and o.get("a"): items.append(o)
    seen, per, kept = set(), {}, []
    for o in sorted(items, key=lambda o: len(clean_a(o["a"]) or "x" * 999)):   # shortest facts first
        a = clean_a(o["a"]); lim = 999 if o["repo"] == "curated" else 130 if o["repo"] == "profile" else 20
        if not a or a in seen or per.get(o["repo"], 0) >= lim: continue
        seen.add(a); per[o["repo"]] = per.get(o["repo"], 0) + 1; kept.append(o)
    items = kept
    off = [json.loads(l)["q"] for f in sorted(glob.glob("qa/offtopic*.jsonl")) for l in open(f) if l.strip()]
    return items, off

def make(holdout=0.0, aug=2, seed=0):
    rng = random.Random(seed); items, off = load(); train, test = [], []
    for o in items:
        a = clean_a(o["a"]); qs = [norm_q(q) for q in o["qs"]]
        if holdout and rng.random() < holdout and len(qs) > 2: test.append((qs[-1], a, o["repo"])); qs = qs[:-1]
        for q in qs:
            train.append((q, a))
            for _ in range(aug * (3 if o["repo"] == "curated" else 1)):
                v = typo(rng.choice(PRE) + q, rng)
                if rng.random() < 0.3: v = v.rstrip("?")
                train.append((v[:140], a))
    for q in off:
        q = norm_q(q)
        for _ in range(3): train.append((typo(rng.choice(PRE) + q, rng), rng.choice(REFUSE)))
        if holdout and rng.random() < holdout: test.append((q, "REFUSE", "off"))
    rng.shuffle(train); return train, test

def fmt(q, a): return "q: " + q + "\na: " + a
