# SVG filter pipeline test: result

Question: does GitHub serve a filter-based SVG intact, and render it, when it is used as an image (as in a README)?

Method: pushed a known-good filter network SVG (`filter_test.svg`, 308 KB, a 4-8-4 integer MLP built from feConvolveMatrix and feComponentTransfer primitives) plus padded copies of 1, 4 and 8 MB, then fetched what GitHub serves and rendered it in headless Chrome through an `<img>` tag.

Results (2026-11-08):

| Check | Result |
|---|---|
| Rendered markdown keeps `<img src="x.svg">` (relative path, resolves to github.com/.../raw/main/... then raw.githubusercontent.com, not the camo proxy) | yes |
| Served `Content-Type` | `image/svg+xml` |
| Served bytes vs repo file, 0.3 MB, 1.3 MB, 4.2 MB, 8.2 MB | byte-identical (`cmp`), no size cap hit |
| Chrome headless `<img>` render of the GitHub-served file vs the local file (160x480) | pixel-identical, max difference 0 (also for the 8 MB padded copy) |

Not verified: Firefox and Safari rendering, and a browser rendering inside the live README page itself (headless Chrome loaded the image from a local HTML page). The padded copies were removed after the test; they only checked transport size.
