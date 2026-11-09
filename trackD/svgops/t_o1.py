import sys, time, json
from ops_lib import *
rng = np.random.default_rng(3)
Z = np.load("/Users/jhome/readme-lm/trackD/qat8/q8_params.npz", allow_pickle=False)
def real_weights(name, sin_over_sout=1.0):
    cc = Z[name + "_c"].astype(np.float64); s = Z[name + "_s"].astype(np.float64)
    return (cc.reshape(cc.shape[0], -1, 32) * s[..., None]).reshape(cc.shape)
def run_linear(name, n_out, n_in, H=8, wmax=127, img=False, tag=None):
    Wr = real_weights(name); assert Wr.shape == (n_out, n_in)
    Wr = Wr * (0.1 / Wr.std())      # demo gain: out code per in code (rms 0.1, outputs mostly unsaturated)
    E, d = quantize_weights(Wr, wmax)
    sizes = split_chunks(n_in); chunks = []; col = 0
    for n in sizes: chunks.append((col, n)); col += n + 1
    out0 = col + 1; out_cols = [out0 + j for j in range(n_out)]; Wd = out0 + n_out + 260
    c = Ctx(Wd, H)
    U = np.clip(rng.normal(128, 25, (H, n_in)).round(), 0, 255).astype(np.int64)
    img_arr = np.zeros((H, Wd), np.uint8)
    off = 0
    for (c0, n) in chunks: img_arr[:, c0:c0 + n] = U[:, off:off + n]; off += n
    c.image(img_arr, res="src")
    bacc = rng.integers(-500, 500, n_out)
    res = linear(c, "src", chunks, E, d, 128, 128, bacc, out_cols)
    svg = c.g.svg(); t0 = time.time()
    a, dt = render(svg, Wd, H, tag or "o1", img=img)
    got = a[:, out0:out0 + n_out, 0].astype(np.int64)
    ref = linear_ref(U, sizes, E, d, 128, 128, bacc)
    ok = (got == ref).all()
    if not ok:
        bad=np.argwhere(got!=ref); print("bad cols",sorted(set((bad[:,1]).tolist()))[:40], "rows",sorted(set(bad[:,0].tolist())))
        j=bad[0][1]; print("example",got[bad[0][0],j],ref[bad[0][0],j])
    return dict(op=f"linear {n_in}->{n_out} ({name})", PASS=bool(ok), mismatches=int((got != ref).sum()), of=int(ref.size), prims=c.g.count(), svg_bytes=len(svg), render_s=round(dt, 2), d=d, chunks=sizes, sat_frac=float(((ref == 0) | (ref == 255)).mean()))
if __name__ == "__main__":
    res=[]
    for name, no, ni in [("o0", 128, 128), ("qkv0", 384, 128), ("u0", 512, 128), ("dn0", 128, 512)]:
        r = run_linear(name, no, ni, tag="o1_" + name); print(r, flush=True); res.append(r)
    r = run_linear("o1", 128, 128, tag="o1_img", img=True); r["op"] += " via <img>"; print(r, flush=True); res.append(r)
    json.dump(res, open("results_o1.json", "w"), indent=1)
