"""Training text for the generation demo: coherent sentences about Jerry's work, mapped to the 32-symbol vocabulary."""
import json, glob, re, os
VOC = "abcdefghijklmnopqrstuvwxyz .,-'?"
QA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qa")
BAD = ("stackadapt", "warg", "compiling programs", "model compression")
DIG = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
def vocab_map(s):
    """lowercase; digits spelled out ('3.5' -> 'three point five'); anything else outside the vocabulary -> space; collapse spaces."""
    s = s.lower(); s = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + " point " + m.group(2), s)
    s = re.sub(r"\d", lambda m: " " + DIG[int(m.group(0))] + " ", s)
    s = "".join(ch if ch in VOC else " " for ch in s); s = re.sub(r"\s+", " ", s)
    s = re.sub(r" ([.,?])", r"\1", s); return s.strip()
def sentences(maxlen=88):
    seen_stem = set(); out = []; rank = []
    order = ["g9", "g10", "g8", "g7", "g6", "g5", "g4", "g3", "g2", "g1"]
    for g in order:
        f = f"{QA}/{g}.jsonl"
        if not os.path.exists(f): continue
        for l in open(f):
            a = json.loads(l)["a"]
            if any(b in a.lower() for b in BAD): continue
            for sent in re.split(r"(?<=[.?])\s+(?=[A-Z])", a):
                t = vocab_map(sent)
                if not (20 <= len(t) <= maxlen) or t[-1] not in ".?": continue
                if t.split(' ')[0] in ('it','its','they','this','these','he','his','that','there','no','and','but','so','then','also','both','each','the'): continue
                if t[:7] in seen_stem: continue
                seen_stem.add(t[:7]); out.append(t); rank.append(0 if g in ('g9', 'g10') else 1 if g in ('g8', 'g7') else 2)
    # keep a maximal subset whose (7-char context -> next char) pairs never contradict each other (otherwise exact recitation is impossible);
    # most distinctive sentences (fewest conflicts with the pool) first, curated files first among equals
    P = []
    for t in out:
        u = " " * 7 + t + " "; pairs = [(u[i - 7:i], u[i]) for i in range(14, len(u))]; d = dict(pairs)
        P.append(d if len(d) == len(set(pairs)) else {"\0": "\0", "\1": "\1"})      # self-contradictory sentence: poison pill (never kept)
    from collections import defaultdict
    owner = defaultdict(set)
    for k, d in enumerate(P):
        for c, n in d.items(): owner[c].add((n, k))
    deg = []
    for k, d in enumerate(P):
        bad = set()
        for c, n in d.items():
            for n2, k2 in owner[c]:
                if n2 != n: bad.add(k2)
        deg.append(len(bad))
    order = sorted(range(len(out)), key=lambda k: (rank[k], deg[k], k)); table = {}; keep = []
    for k in order:
        if "\0" not in P[k] and all(table.get(c, n) == n for c, n in P[k].items()): keep.append(out[k]); table.update(P[k])
    return keep
if __name__ == "__main__":
    s = sentences(); print(len(s), sum(map(len, s))); print("\n".join(s[:8]))
