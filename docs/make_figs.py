#!/usr/bin/env python3
"""Generate the explainer's figures into docs/img/ from measured numbers.
Charts are standalone SVGs (own card background, light/dark via prefers-color-scheme).
The GIF replays a real GPT-2 run (issue reply table: chosen token + top-3 probabilities per step)."""
import json, math, os, re, sys
from html import escape
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "img"); os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(HERE)
CSS = """:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e3e2de;--c1:#2a78d6;--c2:#eb6834}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34342f;--c1:#3987e5;--c2:#d95926}}
text{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:var(--ink)}
.t2{fill:var(--ink2)} .bg{fill:var(--bg)} .grid{stroke:var(--grid);stroke-width:1} .c1{fill:var(--c1)} .c2{fill:var(--c2)}
.s1{stroke:var(--c1)} .s2{stroke:var(--c2)} .ring{fill:var(--bg);stroke:var(--c2);stroke-width:2.5}"""
def head(w, h, title, sub):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">'
            f'<title>{escape(title)}</title><style>{CSS}</style><rect class="bg" width="{w}" height="{h}" rx="10"/>'
            f'<text x="24" y="32" font-size="16" font-weight="600">{escape(title)}</text><text class="t2" x="24" y="52" font-size="12">{escape(sub)}</text>')
def save(name, body): open(os.path.join(OUT, name), "w").write(body + "</svg>\n"); print("wrote", name)

# ---------- 1. time per answer (log dot plot) ----------
def fig_time():
    rows = [("numpy engine in an Action (one QA answer)", "m", 0.2, None, "0.2 s"),
            ("XSLT in Chrome 156 (full QA answer)", "m", 163, None, "163 s"),
            ("XSLT with xsltproc (same answer)", "m", 197, None, "197 s"),
            ("Liquid in a Jekyll build", "e", 1020, 1320, "17-22 min"),
            ("CSS + variable fonts, scaled from the blog", "e", 660, 3900, "11-65 min")]
    W, H, L, R, T = 760, 330, 300, 40, 84; pw = W - L - R; lo, hi = math.log10(0.1), math.log10(10000)
    X = lambda v: L + (math.log10(v) - lo) / (hi - lo) * pw
    o = [head(W, H, "Time to produce one QA answer, by where the arithmetic runs", "Log scale. Filled = measured on a laptop. Ringed = extrapolated from measured throughput or published numbers.")]
    for v, lab in [(0.1, "0.1 s"), (1, "1 s"), (10, "10 s"), (100, "100 s"), (1000, "1,000 s"), (10000, "10,000 s")]:
        o.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T-8}" y2="{T+len(rows)*42-14}"/><text class="t2" x="{X(v):.1f}" y="{T+len(rows)*42+4}" font-size="11" text-anchor="middle">{lab}</text>')
    for i, (lab, kind, a, b, txt) in enumerate(rows):
        y = T + i * 42 + 8
        o.append(f'<text x="{L-14}" y="{y+4}" font-size="12.5" text-anchor="end">{escape(lab)}</text>')
        if kind == "e":
            o.append(f'<line class="s2" x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{y}" y2="{y}" stroke-width="4" stroke-linecap="round" opacity="0.55"/>')
            o.append(f'<circle class="ring" cx="{X(a):.1f}" cy="{y}" r="6"/>')
            o.append(f'<text x="{X(b)+12:.1f}" y="{y+4}" font-size="12" font-weight="600">{txt}</text>' if X(b) + 90 < W else f'<text x="{X(a)-12:.1f}" y="{y-10}" font-size="12" font-weight="600" text-anchor="end">{txt}</text>')
        else:
            o.append(f'<circle class="c1" cx="{X(a):.1f}" cy="{y}" r="6.5"/><text x="{X(a)+14:.1f}" y="{y+4}" font-size="12" font-weight="600">{txt}</text>')
    save("time-per-answer.svg", "".join(o))

# ---------- 2. precision (dot plot) ----------
def fig_precision():
    pts = [("8-bit,\nfilter noise", 80.0, 1), ("8-bit", 87.5, 1), ("10-bit", 95.0, 1), ("16-bit", 97.5, 1), ("8-bit,\nretrained", 100.0, 2), ("retrained,\n+-2 level noise", 96.4, 2)]
    W, H, L, R, T, B = 760, 350, 70, 40, 100, 60; pw = W - L - R; ph = H - T - B
    Y = lambda v: T + ph - v / 100 * ph
    o = [head(W, H, "Why 8 bits is the crux for SVG filters", "Share of QA answers exactly equal to the reference, by activation precision.")]
    o.append('<circle class="c1" cx="30" cy="70" r="5"/><text class="t2" x="42" y="74" font-size="12">no retraining (simulation)</text>')
    o.append('<circle class="c2" cx="250" cy="70" r="5"/><text class="t2" x="262" y="74" font-size="12">retrained at 8 bits (measured through the integer reference, 698 questions)</text>')
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="t2" x="{L-10}" y="{Y(v)+4:.1f}" font-size="11" text-anchor="end">{v}%</text>')
    for i, (lab, v, ser) in enumerate(pts):
        x = L + pw * (i + 0.5) / len(pts)
        o.append(f'<line class="s{ser}" x1="{x:.1f}" x2="{x:.1f}" y1="{Y(0):.1f}" y2="{Y(v):.1f}" stroke-width="3" stroke-linecap="round" opacity="0.5"/><circle class="c{ser}" cx="{x:.1f}" cy="{Y(v):.1f}" r="7"/>')
        o.append(f'<text x="{x:.1f}" y="{Y(v)-14:.1f}" font-size="13" font-weight="600" text-anchor="middle">{v:.1f}%</text>')
        for j, line in enumerate(lab.split("\n")): o.append(f'<text class="t2" x="{x:.1f}" y="{H-B+22+j*15}" font-size="12" text-anchor="middle">{escape(line)}</text>')
    save("precision.svg", "".join(o))

# ---------- 3. GPT-2 graph coverage (bars) ----------
def fig_coverage():
    S = json.load(open(os.path.join(ROOT, "graph/gpt2/stats.json")))
    S = sorted(S, key=lambda d: -d["mass_d8"]); W, H, L, R, T = 760, 100 + len(S) * 26, 300, 60, 84; pw = W - L - R
    o = [head(W, H, "How long real GPT-2 text stays inside a 2,000-state link graph", "Chance a sampled continuation is still in the graph after 8 tokens (temperature 0.6).")]
    for v in (0, 25, 50, 75, 100):
        x = L + pw * v / 100; o.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{T-8}" y2="{T+len(S)*26-6}"/><text class="t2" x="{x:.1f}" y="{T+len(S)*26+10}" font-size="11" text-anchor="middle">{v}%</text>')
    for i, d in enumerate(S):
        y = T + i * 26; v = d["mass_d8"] * 100; lab = d["prompt"].replace("Question: ", "Q: ").replace("\nAnswer:", " A:").replace("\n", " ")[:46]; w = max(v / 100 * pw, 3)
        o.append(f'<text x="{L-12}" y="{y+11}" font-size="12" text-anchor="end">{escape(lab)}</text><rect class="c1" x="{L}" y="{y}" width="{w:.1f}" height="14" rx="4"/><text x="{L+w+8:.1f}" y="{y+11}" font-size="12" font-weight="600">{v:.1f}%</text>')
    save("gpt2-graph-coverage.svg", "".join(o))

# ---------- 4. QA graph compression (two bars) ----------
def fig_compress():
    s = json.load(open(os.path.join(ROOT, "graph/qa/stats.json"))); a, b = s["states_raw"], s["states_after_chain_contraction"]; nq = s["questions"]
    W, H, L, R, T = 760, 220, 250, 90, 90; pw = W - L - R
    o = [head(W, H, f"QA click-graph size for {nq} questions", "States a reader would have to click through. Runs where the model has a single confident continuation merge into one link.")]
    for i, (lab, v) in enumerate([("one state per character", a), ("after merging single-continuation runs", b)]):
        y = T + i * 46; w = max(v / a * pw, 3)
        o.append(f'<text x="{L-12}" y="{y+12}" font-size="12.5" text-anchor="end">{lab}</text><rect class="c1" x="{L}" y="{y}" width="{w:.1f}" height="16" rx="4"/><text x="{L+w+8:.1f}" y="{y+13}" font-size="13" font-weight="600">{v:,}</text>')
    o.append(f'<text class="t2" x="{L}" y="{T+104}" font-size="12">about {a/b:.0f}x fewer states; every one of the {nq} answers still reaches its end state</text>')
    save("qa-graph-compression.svg", "".join(o))

# ---------- GIF: GPT-2 generating, top-3 per step ----------
def gif():
    from PIL import Image, ImageDraw, ImageFont
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else ""
    rows = []
    for m in re.finditer(r"^\| (\d+) \| `(.*?)` \| ([\d.]+)% \| (.*) \|$", src, re.M):
        top = [(t, int(p)) for t, p in re.findall(r"`(.*?)` (\d+)%", m.group(4))]
        rows.append((m.group(2), float(m.group(3)), top))
    if not rows: print("no rows for gif"); return
    prompt = re.search(r"\*\*Prompt:\*\* `?(.*?)`?\n", src).group(1); mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15); small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13); bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    W, H = 760, 366; BG, INK, MUT, C1, C2 = (13, 17, 23), (230, 237, 243), (139, 148, 158), (57, 135, 229), (217, 89, 38)
    def wrap(s, n=62):
        out, cur = [], ""
        for w in s.split(" "):
            if len(cur) + len(w) + 1 > n: out.append(cur); cur = w
            else: cur = (cur + " " + w).strip() if cur else w
        return out + ([cur] if cur else [])
    frames, text = [], prompt
    def draw(text, step, top, chosen, chosen_p=0.0):
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        d.text((20, 14), "GPT-2 small (124M params, int8), run from Markdown: one frame per token", font=bold, fill=INK)
        d.text((20, 36), "text so far", font=small, fill=MUT)
        y = 62
        for line in wrap(text.replace("\n", " ")): d.text((20, y), line, font=mono, fill=INK); y += 21
        if top:
            d.text((20, 188), f"step {step}: GPT-2's top-3 next tokens (blue = the sampled one)", font=small, fill=MUT)
            bars = list(top)
            if chosen is not None and chosen not in [t for t, _ in top]: bars.append((chosen, max(int(round(chosen_p)), 1)))
            for i, (t, p) in enumerate(bars):
                yy = 214 + i * 36; vis = repr(t)[1:-1].replace(" ", "␣") if False else t.replace(" ", "·").replace("\n", "\\n")
                col = C1 if t == chosen else (90, 99, 110)
                d.text((20, yy + 2), f"{vis:>10}", font=mono, fill=INK); d.rounded_rectangle((130, yy, 130 + max(p, 1) / 100 * 560, yy + 22), radius=5, fill=col); d.text((130 + max(p, 1) / 100 * 560 + 10, yy + 3), f"{p}%", font=small, fill=INK)
        return im
    frames.append(draw(text, 0, [], None))
    for i, (tok, p, top) in enumerate(rows):
        frames.append(draw(text, i + 1, top, tok, p)); text += tok
        frames.append(draw(text, i + 1, top, tok, p))
    frames += [draw(text, len(rows), [], None)] * 8
    dur = [900] + [260, 520] * len(rows) + [350] * 8
    pal = [f.quantize(colors=48, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(os.path.join(OUT, "gpt2-generation.gif"), save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=1)
    print("wrote gpt2-generation.gif", len(frames), "frames", os.path.getsize(os.path.join(OUT, "gpt2-generation.gif")) // 1024, "KB")


# ---------- architecture diagrams (hand-laid boxes and arrows; no Mermaid so they render everywhere) ----------
DCSS = CSS + """.node{fill:var(--node);stroke:var(--grid);stroke-width:1.5} .hl{stroke:var(--c1);stroke-width:2} .arr{stroke:var(--ink2);stroke-width:1.6;fill:none} .dash{stroke:var(--ink2);stroke-width:1.4;fill:none;stroke-dasharray:5 4} .ah{fill:var(--ink2)} .cap{fill:var(--ink2)} .ttl{font-weight:600}
:root{--node:#f3f2ee}@media (prefers-color-scheme: dark){:root{--node:#262624}}"""
def flow(name, title, sub, rows, W=860, nh=66, gap=34, hl=(), notes=()):
    """rows: list of rows; a row is a list of node labels ('\\n' = line break). Arrows run left to right in a row and wrap to the next row.
       hl: set of (row, col) boxes to outline in the accent color. notes: [(x, y, text)] small captions."""
    top = 78; rg = 52; H = top + len(rows) * nh + (len(rows) - 1) * rg + 56
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title><style>{DCSS}</style><rect class="bg" width="{W}" height="{H}" rx="10"/>',
         f'<text class="ttl" x="24" y="32" font-size="16">{escape(title)}</text><text class="t2" x="24" y="52" font-size="12">{escape(sub)}</text>',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>']
    pos = []
    for r, row in enumerate(rows):
        n = len(row); nw = (W - 48 - (n - 1) * gap) / n; y = top + r * (nh + rg); rp = []
        for c, lab in enumerate(row):
            x = 24 + c * (nw + gap); rp.append((x, y, nw))
            o.append(f'<rect class="node{" hl" if (r, c) in hl else ""}" x="{x:.1f}" y="{y}" width="{nw:.1f}" height="{nh}" rx="9"/>')
            lines = lab.split("\n"); y0 = y + nh / 2 - (len(lines) - 1) * 8 + 4
            for k, t in enumerate(lines): o.append(f'<text x="{x+nw/2:.1f}" y="{y0+k*16:.1f}" font-size="12.5" text-anchor="middle">{escape(t)}</text>')
            if c: o.append(f'<line class="arr" x1="{x-gap+2:.1f}" y1="{y+nh/2}" x2="{x-3:.1f}" y2="{y+nh/2}" marker-end="url(#ah)"/>')
        pos.append(rp)
    for r in range(len(rows) - 1):
        x1, y1, w1 = pos[r][-1]; x2, y2, w2 = pos[r + 1][0]; ym = y1 + nh + rg / 2
        o.append(f'<path class="arr" d="M{x1+w1/2:.1f},{y1+nh} L{x1+w1/2:.1f},{ym} L{x2+w2/2:.1f},{ym} L{x2+w2/2:.1f},{y2-3}" marker-end="url(#ah)"/>')
    for (x, y, t) in notes: o.append(f'<text class="cap" x="{x}" y="{y}" font-size="12">{escape(t)}</text>')
    save(name, "".join(o))

def fig_diagrams():
    flow("arch-fast-version.svg", "The fast version, end to end", "A GitHub Action only does I/O; the extracted engine does the arithmetic.",
         [["Viewer opens an issue\ngpt2: ...  or  qa: ...", "GitHub issue", "GitHub Action runner\n(only does I/O)"],
          ["mdrt/GPT2.md or QA.md\nengine + program + tokenizer\nare all Markdown", "extracted engine runs the\nforward pass in numpy\n(reads the weight shards)", "rendered Markdown\n+ typing animation", "issue reply and\nREADME image"]],
         hl={(1, 1)}, notes=[(24, 78+2*66+52+44, "Weights are Markdown too: gpt2/*.md shards and qa/weights.md.")])
    flow("arch-qa-transformer.svg", "Inside the QA transformer", "600K parameters, character-level, 4-bit weights. Small enough to live inside a README.",
         [["tokens\n(86-character\nvocabulary)", "embedding 86x128\n(shared with the\noutput head)", "3 identical\nblocks (below)", "final\nRMSNorm", "tied head\n128 to 86", "argmax:\nthe next\ncharacter"],
          ["RMSNorm", "linear\n128 to 384\n(q, k, v)", "RoPE on q, k\ncausal attention\n4 heads x 32", "linear\n128 to 128\n+ residual", "RMSNorm\nlinear\n128 to 512", "GELU", "linear\n512 to 128\n+ residual"]],
         hl={(0, 2)}, notes=[(24, 78+2*66+52+44, "Each block is the row above: attention sub-block (left four) then feed-forward sub-block (right three), each added back to the stream.")])
    # where the arithmetic can happen: three columns
    W, H = 860, 300; cols = [("Fast version", ["GitHub Action runner", "numpy engine extracted", "from a Markdown document", "", "real GPT-2 and the QA model"], 0),
                              ("Pure markup (secondary build)", ["GitHub's MediaWiki renderer:", "{{#expr}} in the page", "WORKS (toy, verified live)", "", "SVG filters in the viewer's browser:", "every op exact in Chrome; the full", "chain does not render yet"], 1),
                              ("Probed, not shipped", ["XSLT: works, Chrome removes it", "on 2026-11-17", "Liquid in Jekyll: exact but slow", "CSS + fonts: feasible, slowest,", "last resort"], 0)]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Where the arithmetic can happen"><title>Where the arithmetic can happen</title><style>{DCSS}</style><rect class="bg" width="{W}" height="{H}" rx="10"/>',
         '<text class="ttl" x="24" y="32" font-size="16">Where the arithmetic can happen</text><text class="t2" x="24" y="52" font-size="12">Three places we ran a language model, by who does the multiply-accumulates.</text>']
    cw = (W - 48 - 2 * 20) / 3
    for i, (head, lines, h) in enumerate(cols):
        x = 24 + i * (cw + 20)
        o.append(f'<rect class="node{" hl" if h else ""}" x="{x:.1f}" y="76" width="{cw:.1f}" height="196" rx="10"/><text class="ttl" x="{x+16:.1f}" y="104" font-size="14">{escape(head)}</text>')
        for k, t in enumerate(lines): o.append(f'<text x="{x+16:.1f}" y="{130+k*21}" font-size="12.5">{escape(t)}</text>')
    save("arch-where-arithmetic.svg", "".join(o))
    flow("arch-mediawiki-net.svg", "How the pure-markup net works", "GitHub's own renderer evaluates every expression when the page loads.",
         [["trained weights\n(small integers)", "gen_mediawiki.py\nunrolls every neuron\ninline", "README.mediawiki\n{{#expr: ...}}\neverywhere"],
          ["GitHub's renderer\nevaluates every\nexpression at render time", "table: predicted next\ncharacter per input\ncontext, live", "check_live.py compares\nwith a python reference:\n40/40 exact"]],
         hl={(1, 0)})


def fig_three_versions():
    W, H = 860, 340; cw = (W - 48 - 2 * 20) / 3
    cards = [("A", "Normal", "NEEDS A MINIMAL ACTION", 0, ["real GPT-2 (124M) + a 600K QA model", "weights are Markdown files", "the engine is one Markdown document", "plus one line of Python on an Action", "", "issue  ->  Action  ->  animation"]),
             ("B", "Pure MediaWiki", "NO ACTION TO COMPUTE", 1, ["a MediaWiki page of {{#expr}} formulas", "GitHub's own renderer does the math", "verified 40/40 vs a Python reference", "a toy: 2-char context, 12 units", "", "just open the page"]),
             ("C", "Pure SVG", "NO ACTION TO COMPUTE", 1, ["a neural net drawn as SVG filters", "your browser's filter engine computes", "224 ReLU units, bit-exact, no code", "Chrome, 100% zoom, 1x display", "", "just open the SVG"])]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="The three versions of README-LM"><title>The three versions of README-LM</title><style>{DCSS}:root{{--g:#1a7f37}}@media (prefers-color-scheme: dark){{:root{{--g:#3fb950}}}} .gs{{stroke:var(--g);stroke-width:2}} .os{{stroke:var(--c2);stroke-width:2}} .gf{{fill:var(--g)}} .of{{fill:var(--c2)}} .wt{{fill:#fff;font-weight:700}}</style><rect class="bg" width="{W}" height="{H}" rx="10"/>',
         '<text class="ttl" x="24" y="32" font-size="16">Three versions, one idea: how much of the arithmetic can a document renderer do?</text>',
         '<text class="t2" x="24" y="52" font-size="12">In the live repo an Action also takes your typed text for B and C; opening the page or the SVG itself needs no Action.</text>']
    for i, (tag, name, badge, green, lines) in enumerate(cards):
        x = 24 + i * (cw + 20); st = "gs" if green else "os"; fl = "gf" if green else "of"
        o.append(f'<rect class="node {st}" x="{x:.1f}" y="72" width="{cw:.1f}" height="246" rx="12"/>')
        o.append(f'<rect class="{fl}" x="{x+14:.1f}" y="86" width="{cw-28:.1f}" height="24" rx="12"/><text class="wt" x="{x+cw/2:.1f}" y="103" font-size="11" text-anchor="middle">{badge}</text>')
        o.append(f'<text class="ttl" x="{x+16:.1f}" y="140" font-size="20">{tag} · {escape(name)}</text>')
        for k, t in enumerate(lines): o.append(f'<text x="{x+16:.1f}" y="{170+k*22}" font-size="12.5"{" font-weight=\"600\"" if k==len(lines)-1 else ""}>{escape(t)}</text>')
    save("arch-three-versions.svg", "".join(o))

if __name__ == "__main__":
    fig_time(); fig_precision(); fig_coverage(); fig_compress(); fig_diagrams(); fig_three_versions(); gif()
