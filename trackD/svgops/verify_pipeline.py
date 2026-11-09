import sys, json, time
from svg_eval import *
import core
TMO = 240
def run(T, stops, prompt_q="what is x", log=None):
    p = make_prompt(T, prompt_q[:T - 7]); ids = mk(p); pl = build(T, [ids]); tr = forward_ref(pl, ids)
    ref_full = pl.m.forward_token(ids)
    out = []
    for stop in stops:
        t0 = time.time()
        svg, view, c = build_svg(pl, ids, stop=None if stop == "glyph" else stop); tb = time.time() - t0
        x, y, w, h = view
        try: a, dt = core.render_pw(svg, w, h, f"v{T}_{stop.replace('.','_')}", timeout=TMO)
        except TimeoutError as e:
            r = dict(stage=stop, PASS=None, note=str(e), prims=c.g.count(), svg_bytes=len(svg)); print(r, flush=True); out.append(r)
            if log: json.dump(out, open(log, "w"), indent=1)
            continue
        if stop == "glyph":
            atlas = make_atlas(list(Q.VOC) + ["EOS"], GW, GH, fsize=int(GH * 0.72)); i = int(tr["argmax"][-1])
            ref = atlas[:, i * GW:(i + 1) * GW]; got = a[:, :, 0]
        elif stop == "argmax": got = a[:, 127, 0].astype(np.int64); ref = tr["argmax"]
        else:
            got = a[:, :, 0].astype(np.int64); ref = tr[stop] if stop in tr else None
        if stop.endswith((".q", ".qr", ".ao")):
            got = got[:, ACOLS]
        if stop.endswith(".g"):
            uc = [j + j // 171 for j in range(512)]; got = got[:, uc]; ref = tr[stop]
        ok = None if ref is None else bool((got == ref).all())
        r = dict(stage=stop, PASS=ok, mism=None if ref is None else int((got != ref).sum()), of=None if ref is None else int(np.size(ref)), prims=c.g.count(), svg_bytes=len(svg), render_s=round(dt, 1))
        if stop == "glyph": r["glyph_char"] = (Q.VOC + "<EOS>")[i] if i < 86 else "EOS"
        print(r, flush=True); out.append(r)
        if log: json.dump(out, open(log, "w"), indent=1)
    return out, tr, ref_full, pl
if __name__ == "__main__":
    T = int(sys.argv[1]); stops = sys.argv[2].split(",")
    run(T, stops, log=f"results_pipe_T{T}.json")
