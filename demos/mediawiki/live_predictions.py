#!/usr/bin/env python3
"""live_predictions.py RENDERED.html : read the next-character predictions that GitHub's renderer computed.
Input: the HTML GitHub's API returns for the generated MediaWiki page (every number in its tables is evaluated by
GitHub's MediaWiki renderer; this script only reads them). Prints JSON {"windows":[{"w","actual","pred"}], "text": "..."}.
In 'text' a space is shown as a middle dot so it survives SVG text."""
import sys, re, json
from html.parser import HTMLParser
class T(HTMLParser):
    def __init__(s): super().__init__(); s.tables = []; s.row = None; s.cell = None
    def handle_starttag(s, t, a):
        if t == "table": s.tables.append([])
        if t == "tr": s.row = []; s.tables[-1].append(s.row)
        if t in ("td", "th"): s.cell = ""
    def handle_endtag(s, t):
        if t in ("td", "th") and s.cell is not None: s.row.append(s.cell.strip()); s.cell = None
    def handle_data(s, d):
        if s.cell is not None: s.cell += d
html = open(sys.argv[1]).read()
if "Expression error" in html or "{{#" in re.sub(r"<code>.*?</code>", "", html, flags=re.S): sys.exit("page not evaluated by the renderer")
p = T(); p.feed(html)
t1 = [tb for tb in p.tables if tb and tb[0] and tb[0][0] == "n"]
if not t1: sys.exit("no window table in the rendered page")
dot = lambda c: "·" if c in ("␣", " ", "") else c
wins = []
for r in t1[0][1:]:
    w = r[1]; pred = r[-1]; actual = r[2] if len(r) >= 4 else ""
    wins.append({"w": w, "actual": actual, "pred": pred})
text = "  ".join(f"{dot(w['w'][0])}{dot(w['w'][1])}->{dot(w['pred'])}" for w in wins)
print(json.dumps({"windows": wins, "text": text}, ensure_ascii=False))
