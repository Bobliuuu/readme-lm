#!/usr/bin/env python3
"""Stage 3 (stretch): how deep can a chain of per-neuron convolve layers get in Chrome with this builder (shared shifted node per
layer, one merge + LUT per layer)?  Random integer MLP, width WD, depth D, N window rows (RH=1, compute only); each run is also
checked bit-exact against the numpy integer mirror.   python3 demos/svg/stage3.py [timeout_s]"""
import os, sys, json, time, tempfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import svgnn, harness

def linear_tight(c, src, c0, n, E, d, zin, zout, bacc, out_cols):
    """same as svgnn.linear but every node is confined to the exact sub-rectangle it needs (tight primitive subregions)."""
    x0 = out_cols[0]; sh = c.offset(src, x0 - c0, 0, x=x0, y=0, w=n + 1, h=c.rows); outs = []
    for j in range(E.shape[0]):
        outs.append(svgnn.conv_neuron(c, sh, n, out_cols[j], x0, E[j], d, int(bacc[j]) - zin * int(E[j].sum()), zout))
    return c.merge_at(outs, x0, 0, n + 1, c.rows)

def run(WD, D, N, seed=0, timeout=60, tight=False):
    rng = np.random.default_rng(seed); n = WD; CB = 8; W = CB + n + 8 + n + 8
    c = svgnn.Ctx(W, N)
    x0 = np.clip(rng.normal(128, 40, (N, n)).round(), 0, 255).astype(np.uint8)
    cur = c.merge([c.image(x0, x=CB, y=0)]); U = x0.astype(np.int64); col = CB
    relu = np.array([max(v - 128, 0) * 2 for v in range(256)]).clip(0, 255)
    for k in range(D):
        E = np.clip(np.round(rng.normal(0, 25, (n, n))), -127, 127).astype(np.int64); b = np.zeros(n, np.int64)
        out_col = CB + n + 8 if col == CB else CB          # ping-pong between two column blocks
        lin = linear_tight if tight else svgnn.linear
        p = lin(c, cur, col, n, E, 16, 128, 128, b, list(range(out_col, out_col + n)))
        l = c.lut(p, relu, x=out_col, y=0, w=n, h=N)
        cur = c.merge_at([l], out_col, 0, n + 1, N) if tight else c.merge([l]); col = out_col
        U = relu[svgnn.linear_ref(U, E, 16, 128, 128, b)]
    c.merge([cur])
    svg = svgnn.svg_wrap(c, "".join(c.p), W, N, 0, "", "depth")
    p = os.path.join(tempfile.gettempdir(), "svg_depth_test.svg"); open(p, "w").write(svg)
    try: a, dt = harness.render(p, W, N, timeout=timeout)
    except Exception as e:
        os.remove(p); return dict(width=WD, depth=D, rows=N, prims=c.n, svg_kb=len(svg) // 1024, tight=tight, render_s=None, exact=None, note=str(e)[:60])
    os.remove(p); got = a[:, col:col + n, 0].astype(np.int64)
    return dict(width=WD, depth=D, rows=N, prims=c.n, svg_kb=len(svg) // 1024, tight=tight, render_s=round(dt, 2), exact=f"{int((got == U).sum())}/{U.size}")

if __name__ == "__main__":
    import subprocess
    if len(sys.argv) > 1 and sys.argv[1] == "--one":          # one configuration per process: a crashed renderer cannot poison the next run
        WD, D, tight, to = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5])
        print(json.dumps(run(WD, D, 32, timeout=to, tight=tight)), flush=True); sys.exit(0)
    to = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for tight in (0, 1):
        for WD in (32, 64, 128):
            for D in (1, 2, 3, 4, 5, 6, 8):
                out = subprocess.run([sys.executable, __file__, "--one", str(WD), str(D), str(tight), str(to)], capture_output=True, text=True).stdout.strip().splitlines()
                r = json.loads(out[-1]) if out else dict(width=WD, depth=D, tight=bool(tight), render_s=None, note="process died")
                print(json.dumps(r), flush=True)
                if r["render_s"] is None or r["render_s"] > to / 4: break
