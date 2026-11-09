#!/usr/bin/env python3
"""render_svg.py OUT.svg PROMPT TEXTFILE : typing animation of GPT-2's output (reuses gen.svg)."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gen
out, prompt, tf = sys.argv[1:4]
text = re.sub(r"\s+", " ", open(tf).read()).strip()
open(out, "w").write(gen.svg(prompt[:120], text, "", "GPT-2 small · 124M params · int8 · weights read from Markdown shards"))
