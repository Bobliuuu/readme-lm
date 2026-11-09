#!/usr/bin/env python3
"""check_live.py : verify GitHub's OWN rendering of README.mediawiki (production github-markup/WikiCloth).
   gh api -H 'Accept: application/vnd.github.html+json' repos/Bobliuuu/readme-lm/readme/trackD/mediawiki > out_live.html
   then compares every live cell with the python-int forward pass using check.py's parser/reference."""
import re, os
here = os.path.dirname(os.path.abspath(__file__))
ns = {"__name__": "chk", "__file__": os.path.join(here, "check.py")}
exec(open(os.path.join(here, "check.py")).read().split("good = check")[0], ns)
html = open(os.path.join(here, "out_live.html")).read()
print("Expression error present:", "Expression error" in html, "| unevaluated {{# outside code:", "{{#" in re.sub(r"<code>.*?</code>", "", html))
ns["render"] = lambda mode: (html, 0.0, "live")
print("LIVE GITHUB RENDER ALL EXACT" if ns["check"]("live") else "LIVE GITHUB RENDER MISMATCH")
