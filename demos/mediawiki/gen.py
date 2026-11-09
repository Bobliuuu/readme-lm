#!/usr/bin/env python3
"""Demo 2 generator. Action-side job: sanitize the viewer's text and write it into the page.
All multiply-accumulates, ReLUs, sign tests and decodes are {{#expr}}/{{#ifeq}}/{{#switch}}
evaluated by the markup renderer.  Usage: gen.py "<input text>" -o README.mediawiki"""
import sys, os, re, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
VOC = "abcdefghijklmnopqrstuvwxyz .,-'?"
MAX_BYTES = 420000
M = {k: v for k, v in np.load(os.path.join(HERE, "model.npz")).items()}
H, D = M['W1'].shape[0], M['E'].shape[1]; NB = M['V'].shape[0]; V = len(VOC)

def sanitize(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = "".join(c for c in s if c in VOC)
    s = re.sub(r" +", " ", s).strip()
    return s

def ent(c):  # every displayed char as a numeric entity: no markup can be formed from input
    return "&#9251;" if c == " " else "&#%d;" % ord(c)

def ref_forward(c1, c2):
    x = [int(v) for v in M['E'][c1]] + [int(v) for v in M['E'][c2]]
    h = [max(0, sum(int(M['W1'][j][i]) * x[i] for i in range(2*D)) + int(M['b1'][j])) for j in range(H)]
    z = [sum(int(M['V'][k][j]) * h[j] for j in range(H)) + int(M['b2'][k]) for k in range(NB)]
    return x, h, z

# ---- expression emission (weights + embedding literals are generator-written) ----
def hidden_PN(j, c1, c2):
    x = [int(v) for v in M['E'][c1]] + [int(v) for v in M['E'][c2]]
    P = []; N = []
    for i in range(2*D):
        w = int(M['W1'][j][i]); xi = x[i]
        if w == 0 or xi == 0: continue
        (P if (w > 0) == (xi > 0) else N).append(f"{abs(w)}*{abs(xi)}")
    b = int(M['b1'][j])
    if b > 0: P.append(str(b))
    elif b < 0: N.append(str(-b))
    return "+".join(P) or "0", "+".join(N) or "0"

def relu_expr(P, N):
    return "{{#ifeq:{{#expr:%s>%s}}|1|{{#expr:%s-(%s)}}|0}}" % (P, N, P, N)

def logit_PN(k, c1, c2):
    P = []; N = []
    for j in range(H):
        v = int(M['V'][k][j])
        if v == 0: continue
        hp, hn = hidden_PN(j, c1, c2)
        (P if v > 0 else N).append(f"{abs(v)}*{relu_expr(hp, hn)}")
    b = int(M['b2'][k])
    if b > 0: P.append(str(b))
    elif b < 0: N.append(str(-b))
    return "+".join(P) or "0", "+".join(N) or "0"

SWITCH_CASES = "|".join("".join(str((t >> (NB-1-k)) & 1) for k in range(NB)) + "=" + ent(VOC[t]) for t in range(V))

def next_char_expr(c1, c2):
    parts = []
    for k in range(NB):
        P, N = logit_PN(k, c1, c2)
        parts.append("{{#expr:%s>%s}}" % (P, N))
    return "{{#switch:" + "".join(parts) + "|" + SWITCH_CASES + "}}"

def build(text, orig_len, truncated, fallback):
    ids = [VOC.index(c) for c in text]
    head = []
    w = head.append
    w("= A neural network computed by the page renderer =\n")
    w("'''What this page is.''' A plain MediaWiki file with no JavaScript and no program running when you read it. "
      "A character model (2 characters in, 12 ReLU hidden units, 5 output bits, 32 symbols; &#9251; = space) predicts the next character for every 2-character window of the text below. "
      "The text was taken from the title of a GitHub issue.\n")
    w("== Who computes what ==\n"
      "* '''Written by the generator script''' (it only writes text into this file): the trained integer weights as literals, the embedding-vector numbers of the input characters as literals, and the input characters themselves after sanitizing (lowercase, vocabulary only).\n"
      "* '''Computed by GitHub's renderer when you open this page''': every multiply-accumulate of both layers (<code>#expr</code>), every ReLU (<code>#ifeq</code> on a <code>#expr</code> comparison), every output sign test (<code>#expr</code> comparison) and the 5-bit decode to a character (<code>#switch</code>). The generator never evaluates the network to write a result into the page; no cell below holds a precomputed prediction.\n"
      "* If the renderer did not compute, the prediction cells would show raw <code>&#123;&#123;#switch...</code> text instead of a letter.\n")
    inp = "".join(ent(c) for c in text)
    note = ""
    if fallback: note = " (the submitted text had no usable characters, so a default was used)"
    w(f"Input after sanitizing{note}: <b>{inp}</b> ({len(text)} characters).")
    if truncated:
        w(f"''The input was truncated: only the first {len(text)} of {orig_len} sanitized characters are used, which keeps this page under 420 KB.''")
    w("")
    return head, ids

def tail_section(ids, i):
    a, b = ids[i], ids[i+1]
    nxt = VOC[ids[i+2]] if i+2 < len(ids) else None
    x, h, z = None, None, None
    out = []; w = out.append
    w(f"== Inside the network: last window {ent(VOC[a])}{ent(VOC[b])} ==")
    w("Embedding vectors (generator-written literals): x = [" + ", ".join(str(int(v)) for v in list(M['E'][a]) + list(M['E'][b])) + "]. Nothing below is precomputed.\n")
    w("Hidden layer, h = ReLU(W1 x + b1), each cell live:\n")
    w('{| class="wikitable"\n! unit !! live h')
    for j in range(H):
        P, N = hidden_PN(j, a, b)
        w("|-"); w(f"| h{j+1} || {relu_expr(P, N)}")
    w("|}\n")
    w("Output layer, z = V h + b2 (hidden units are recomputed inline in every cell; the markup has no variables), then bit = 1 if z is greater than 0, else 0:\n")
    w('{| class="wikitable"\n! bit !! live logit z !! live bit')
    for k in range(NB):
        P, N = logit_PN(k, a, b)
        w("|-"); w("| z%d || {{#expr:%s-(%s)}} || {{#expr:%s>%s}}" % (k+1, P, N, P, N))
    w("|}\n")
    w(f"Decoded next character (live): <b>{next_char_expr(a, b)}</b>\n")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    full = sanitize(a.text)
    fallback = False
    if len(full) == 0:
        full = "hello"; fallback = True
    if len(full) == 1: full = " " + full   # need one 2-char window
    orig_len = len(full)
    # greedy: keep first N windows while the page stays <= MAX_BYTES
    def assemble(text, truncated):
        ids = [VOC.index(c) for c in text]
        head, _ = build(text, orig_len, truncated, fallback)
        nw = len(ids) - 1
        rows = ['== Live next-character predictions ==',
                'One row per consecutive 2-character window. The last column is evaluated by the renderer; "actual next" is just the following input character (blank for the last window).\n',
                '{| class="wikitable"\n! n !! window !! actual next !! predicted next (live)']
        for n in range(nw):
            c1, c2 = ids[n], ids[n+1]
            act = ent(VOC[ids[n+2]]) if n+2 < len(ids) else ""
            rows += ["|-", f"| {n+1} || {ent(VOC[c1])}{ent(VOC[c2])} || {act} || {next_char_expr(c1, c2)}"]
        rows.append("|}\n")
        return "\n".join(head) + "\n" + "\n".join(rows) + "\n" + tail_section(ids, nw-1), nw
    text = full; truncated = False
    page, nw = assemble(text, False)
    if len(page.encode()) > MAX_BYTES:
        lo, hi = 2, len(full)   # largest length whose page fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            p, _ = assemble(full[:mid], True)
            if len(p.encode()) <= MAX_BYTES: lo = mid
            else: hi = mid - 1
        text = full[:lo]; truncated = True
        page, nw = assemble(text, True)
    with open(a.out, "w", encoding="utf-8") as f: f.write(page)
    print(json.dumps({"input": text, "windows": nw, "bytes": len(page.encode())}))

main()
