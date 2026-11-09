import sys, json
from ops_lib import *
import core
def run(T, tag="gath"):
    m, P = pick_pitch(T); TT = T * P; H = TT + 2 * P
    rng = np.random.default_rng(1); X = rng.integers(0, 256, (T, 128))
    W = 200; c = Ctx(W, H); arr = np.zeros((T, W), np.uint8); arr[:, :128] = X; c.image(arr, res="s0"); s = c.merge(["s0"], h=T)
    st = gather_stretch(c, s, T, P, m, TT)
    stm = c.merge([st], h=H)
    # compaction test: place values at rows P*t+P-1 : use the stretched image itself (all rows of block t hold X[t]) -> compact must return X
    cp = gather_compact(c, stm, T, P, m, TT)
    c.prim(f'<feComponentTransfer in="{cp}"/>')
    svg = c.g.svg().replace(f'width="{W}" height="{H}" viewBox="0 0 {W} {H}"', f'width="128" height="{T}" viewBox="0 0 128 {T}"', 1)
    a, dt = core.render_pw(svg, 128, T, tag); got = a[:, :128, 0].astype(int)
    # also stretched check: separate render
    c2 = Ctx(W, H); c2.image(arr, res="s0"); s2 = c2.merge(["s0"], h=T); st2 = gather_stretch(c2, s2, T, P, m, TT); c2.prim(f'<feComponentTransfer in="{st2}"/>')
    svg2 = c2.g.svg().replace(f'width="{W}" height="{H}" viewBox="0 0 {W} {H}"', f'width="128" height="{TT}" viewBox="0 0 128 {TT}"', 1)
    a2, dt2 = core.render_pw(svg2, 128, TT, tag + "s"); got2 = a2[:, :128, 0].astype(int); ref2 = np.repeat(X, P, axis=0)
    return dict(T=T, m=m, P=P, compact_roundtrip_ok=bool((got == X).all()), stretch_ok=bool((got2 == ref2).all()), mism_c=int((got != X).sum()), mism_s=int((got2 != ref2).sum()), prims=c.g.count(), t=round(dt, 2))
if __name__ == "__main__":
    for T in (8, 16, 32, 64): print(run(T), flush=True)
