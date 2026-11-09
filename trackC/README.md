# Track C: render-time compute substrates

These are probe results from agents. Everything was emulated locally (headless Chrome, xsltproc, Jekyll, github-markup gems). Behaviour on live github.com or GitHub Pages is unverified unless a report below says otherwise. Each folder has a `REPORT.md` with the full details and the scripts that produced the numbers.

| Rank | Substrate | Folder | One-line verdict |
|---|---|---|---|
| 1 | SVG filters | [svgfilter](svgfilter/REPORT.md) | Works as a README `<img>`; exact 8-bit arithmetic (4-8-4 MLP exact on 24/24 nets); a real model is plausible with 8-bit QAT. |
| 2 | MediaWiki `#expr` | [markups](markups/REPORT.md) | Renders in GitHub's WikiCloth path; toy matvec only; no variables or loops; GitHub-unverified. |
| 3 | Liquid / Jekyll | [liquid](liquid/REPORT.md) | Runs on Pages; exact arithmetic; about 82-115K MAC/s, so 17-22 min per answer against the 10 min deployment limit. |
| 4 | XSLT | [xslt](xslt/REPORT.md) | Full real-model answer in 163-197 s, but Chrome removes XSLT on 2026-11-17 and there is no URL input. |
| 5 | CSS + variable fonts | [cssfont](cssfont/REPORT.md) | Feasible but 11-65 min per answer and 3-4 weeks of work; the reference repo has no license, so never copy its code. |

CSS is last by project priority, not only by feasibility.

Large generated artifacts (model.xml, generated fonts, big SVGs, Jekyll `_site/` output, scratch `work/` dirs, screenshots) are not committed; regenerate them with the scripts.
