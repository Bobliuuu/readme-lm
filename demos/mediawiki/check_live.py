#!/usr/bin/env python3
"""Verify GitHub's OWN rendering of demos/mediawiki/README.mediawiki.
  gh api -H 'Accept: application/vnd.github.html+json' repos/Bobliuuu/readme-lm/readme/demos/mediawiki > out_live.html
  python3 check_live.py out_live.html
Compares every live cell (all window predictions + the worked example) with an independent python-int forward pass."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import verify_html
html = open(sys.argv[1] if len(sys.argv) > 1 else "out_live.html", encoding="utf-8").read()
ok, tot, bad, text = verify_html(html, "live")
print(f"windows' input text reconstructed from page: {text!r}")
print(f"Expression error present: {'Expression error' in html} | unevaluated '{{{{#' outside code: {'{{#' in re.sub(r'<code>.*?</code>', '', html, flags=re.S)}")
print(f"live cells exact: {ok}/{tot} ({100*ok/max(tot,1):.2f}%) {bad or ''}")
print("LIVE GITHUB RENDER ALL EXACT" if tot and ok == tot and not bad else "LIVE GITHUB RENDER MISMATCH")
sys.exit(0 if tot and ok == tot and not bad else 1)
