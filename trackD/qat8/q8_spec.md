# q8 spec: README-LM QA transformer with 8-bit activations everywhere

Normative reference: `q8_ref.py` (pure numpy, int64 arithmetic). This document describes exactly what it computes.
If the two ever disagree, the code wins. Constants: `q8_params.npz` (weights + calibrated formats) and `q8_tables.npz`
(every LUT and integer multiplier already derived, so a generator does not have to re-derive anything).

## 0. Conventions

* Model: d=128, H=4 heads, hd=32, 3 layers, FFN=512, V=86 (85 chars + EOS=85), char vocab `VOC` from `qa_model.py`.
  RoPE base 10000, RMSNorm eps 1e-5, GELU = exact erf GELU, tied embeddings. Same weights layout as `qa_model.py`.
* **Tensor code**: every activation tensor element is one uint8 `u` in 0..255. A *format* is `(s, z)`: `real = (u - z) * s`.
  Signed tensors use `z = 128`. Tensors with a skewed range use another integer `z` (GELU out, scores, ...). All `s`,`z` are in
  `q8_params.npz` key `spec` (JSON, keys like `L0.n1.s`, `L0.n1.z`); `z` values and every integer below are what an SVG generator uses.
  The scales are STATIC (calibrated, then QAT fine-tuned); nothing is data dependent except the values themselves.
* `rs(a) = (a + 2^31) >> 32` (arithmetic shift = floor division, i.e. round half up) of an int64 accumulator.
  `clip8(x) = min(max(x,0),255)`. `fx(r) = floor(r * 2^32 + 0.5)` turns a real ratio into a 32-bit fixed-point multiplier `M`.
  Typical op: `out = clip8( rs( acc * M ) + z_out )`. In an SVG graph this is ONE primitive with real coefficient `M / 2^32`
  and offset `z_out`; the primitive's own round+clamp to 0..255 is `rs`+`clip8`. (Chrome's convolve adds +-1..2 level noise on
  fractional kernels, see REPORT.md; the reference is the noiseless definition.)
* Every primitive output is therefore uint8; there is no wider intermediate than one primitive's internal float accumulate.
  Where the reference sums products or terms inside one `rs(...)` the generator may use ONE feConvolveMatrix (<=256 taps) /
  feColorMatrix (<=4 inputs) / feComposite arithmetic (2 inputs). Where a spec step says "8-bit product" the product is
  materialised as its own uint8 tensor before it is summed.
* Weights: 4-bit symmetric codes `c in [-7,7]`, group size 32 along the input dim, fp16 group scale `sw[o,g]`
  (identical storage to the README model; re-quantised from the fine-tuned fp32 weights with `scale = absmax/7 -> fp16`,
  `c = clip(floor(w/scale+0.5),-7,7)`). Weights of qkv/o/u/dn/emb live in `q8_params.npz` as `*_c` (int8 codes) and `*_s` (fp16).
  RMSNorm gains `g1,g2,gf` are fp16 vectors.
* Sequence: `T` tokens, position `p = 0..T-1`, causal. `forward_token(ids)` = logits of the last position.

## 1. Op by op (layer l, input tensor `x` format `X=(sx,zx)`, previous layer output or embedding)

Notation `F(name)` = format of tensor `name` (prefix `L{l}.` omitted).

### 1.1 Embedding
`x0[t, j] = clip8( floor(c_emb[id,j]*sw_emb[id,j//32] / s_emb + 0.5) + 128 )` -> a constant 86x128 uint8 table (`emb_tab`). Format `emb=(s_emb,128)`.

### 1.2 RMSNorm (n1, n2, nf) : input `x` (128 codes), gain `g` (fp16 x128), output format `y=(sy,zy)`
Per-norm constants: `s_sq`, `C` (int), `mm` (real), `sr`, `zr` (int, negative), all in spec as `<norm>.sq.s .C .mm .sr .zr`.
1. `sq[j] = LUT_sq[x[j]]`, `LUT_sq[u] = clip8(floor((u-zx)^2 * sx^2 / s_sq + 0.5))` (unsigned uint8, 256-entry LUT, feComponentTransfer).
2. `S = sum_j sq[j]` (exact integer, <= 32640): one feConvolveMatrix of 128 ones.
3. `ms8 = clip8( rs((S - C) * fx(mm)) )` (one primitive: the convolve's bias carries `-C*mm`).
4. `r8 = LUT_rsq[ms8]`, `LUT_rsq[m] = clip8( floor( 1/sqrt( (m/mm + C)*s_sq/128 + 1e-5 ) / sr + 0.5 ) + zr )`. Real `r = (r8 - zr)*sr` = 1/rms.
5. `y[j] = clip8( rs( (x[j]-zx) * (r8 - zr) * G[j] ) + zy )`, `G[j] = fx( g[j]*sx*sr/sy )` (per-channel int64).
   One feComposite arithmetic per channel: `k1*x*r8 + k2*x + k3*r8 + k4` with `k1=G/2^32`, `k2=-zr*G/2^32`, `k3=-zx*G/2^32`,
   `k4 = zx*zr*G/2^32 + zy` (all /255 normalisations as per REPORT.md).
Note: the sum of squares is made of 8-bit squares (step 1), this is a real precision loss and is modelled.

### 1.3 QKV projection: n1 (format `n1`) -> q, k, v, each (T,128) with their own format `q`,`k`,`v` (signed, z=128)
For output row `o` in 0..383 (o//128 = 0:q, 1:k, 2:v; head h=(o%128)//32, dim d=o%32):
`acc[o] = sum_{g=0..3} M[o,g] * sum_{j in group g} c[o,j]*(n1[j]-zn1)`, `M[o,g] = fx(sw[o,g]*s_n1 / s_out(o))`,
`out[o] = clip8( rs(acc) + z_out(o) )`. One feConvolveMatrix with 128 taps per output neuron, kernel `K_j = c[o,j]*M[o,g(j)]/2^32`
(`group size G_conv = 128`, a single 8-bit rounding per output). Per-group scales are just different kernel numbers.

### 1.4 RoPE (q -> qr, k -> kr; formats `qr`,`kr` signed z=128)
For position `p`, pair `i = 0..15` inside each head (dims 2i, 2i+1), `a1 = q[2i]-zq`, `a2 = q[2i+1]-zq`:
`out[2i]   = clip8( rs( a1*Cr[p,i] - a2*Sr[p,i] ) + 128 )`, `out[2i+1] = clip8( rs( a1*Sr[p,i] + a2*Cr[p,i] ) + 128 )`,
`Cr = fx(cos(p*f_i) * s_q/s_qr)`, `Sr = fx(sin(p*f_i) * s_q/s_qr)`, `f_i = 10000^(-2i/32)`, tables in `q8_tables.npz`
(`L*.qr.C/S`, `L*.kr.C/S`, int64, shape (512,16)). One feComposite arithmetic (two inputs, constants per (p,i)) per output element.

### 1.5 Scores (per head h, query position t, key position t' <= t)
1. 8-bit product tensor (feComposite k1): `pr[t,t',d] = clip8( rs( (qr[t,h,d]-128)*(kr[t',h,d]-128) * Mp ) + 128 )`, `Mp = fx(s_qr*s_kr/s_pr)`; format `pr=(s_pr,128)`.
2. Coarse score (one 32-tap convolve of `pr`): `sc[t,t'] = clip8( rs( sum_d (pr-128) * Msc ) + zsc )`, `Msc = fx(s_pr/sqrt(32)/s_sc)`; format `sc=(s_sc,zsc)` (asymmetric z).
3. Row maximum over t' = 0..t (visible keys only): `m[t] = max_t' sc[t,t']` (exact; tournament of `max(a,b) = b + (a-b)^+`, with `(a-b)^+` one feComposite k2=1,k3=-1 clamped at 0 -- exact for uint8).
4. Fine max-subtracted score: one primitive whose kernel takes the 32 products AND the pixel `m[t]` as one extra tap:
   `dd[t,t'] = clip8( rs( (m[t]-zsc)*Maf - (sum_d (pr-128))*Mbf ) + ZF )`, `Maf = fx(s_sc/s_f)`, `Mbf = fx(s_pr/sqrt(32)/s_f)`,
   `s_f = 9/255`, `ZF = round(s_sc/s_f)` (integer, spec `f.z`). This is "max - score + margin" in fine units (`s_f` nats/level): it lets
   the exp LUT work at 0.035-nat resolution while the stored coarse score may have a much coarser scale.
   (Hardware note: the tap layout puts `m[t]` in the same pixel row as the 32 products; it can be produced by feMerge/feOffset of the
   broadcast max. If that layout is awkward, a 2-primitive equivalent is NOT bit-identical; ask for the variant without the fine stage.)
5. `e = LUT_exp[dd]`, `LUT_exp[u] = clip8(floor(255*exp(-u*s_f) + 0.5))` (masked to 0 for t' > t).
6. `Sg[t] = sum_{t'<=t} e` (exact int up to 255*T; the kernel is the whole visible row; for T > 256 split in 256-tap pieces and add with exact integer feComposite).
7. `S8 = clip8( rs( Sg * fx(1/D) ) )`, `D` = power of two per layer (spec `L*.D`; 32,16,16 in the shipped model). `R8 = LUT_rec[S8]`,
   `LUT_rec[u] = clip8(floor(255/(max(u,1)*D)*256 + 0.5))` (reciprocal, R = R8/256).
8. `p8 = clip8( rs( e * R8 * fx(1/256) ) )`  (= `clip8(floor(e*R8/256 + 0.5))`; exact integer, 256 is a power of two). `p8/255` is the attention probability.
   Maximum value is 254 for a one-hot row.

### 1.6 Attention-weighted values
1. 8-bit product tensor: `pv[t,t',h,d] = clamp( rs( p8[t,t'] * (v[t',h,d]-zv) * Mpv ), -128, 127 )` (stored `+128`), `Mpv = fx(s_v/(255*s_pv))`, format `pv=(s_pv,128)`.
2. Block sum over keys in blocks of KB = 256 (one convolve of <=256 taps each): `part_b = clamp( rs( sum_{t' in block b}(pv) * Mpb ), -128, 127 )`,
   `Mpb = fx(s_pv/s_ao)`. In this build `s_ao = s_pv` exactly so `Mpb = 2^32` (the blocks are plain sums with saturation).
3. `ao[t,h,d] = clip8( 128 + sum_b part_b )`, format `ao=(s_pv,128)`. With T <= 256 there is exactly one block.

### 1.7 Output projection, residual 1
`o = clip8( rs( sum_g Mo[o,g]*sum_{j in g} c[o,j]*(ao[j]-128) ) + zo )` (128 taps, `Mo = fx(sw*s_ao/s_o)`), format `o`.
`r1 = clip8( rs( (x-zx)*fx(sx/s_r1) + (o-zo)*fx(s_o/s_r1) ) + zr1 )` : one feComposite arithmetic (k2,k3,k4), format `r1`.

### 1.8 FFN
`n2 = RMSNorm(r1, g2)` (as 1.2, input format `r1`). `u = clip8(rs(sum_g Mu*sum c*(n2-zn2)) + zu)`: 128 taps -> 512 outputs, format `u`.
`g = LUT_gelu[u]`, `LUT_gelu[a] = clip8(floor(gelu((a-zu)*s_u)/s_g + 0.5) + zg)` (erf GELU in float64), format `g=(s_g,zg)` (asymmetric: GELU is >= -0.17).
Down projection in two halves of 256 inputs (kernel limit): `dp_h = clip8( rs( sum_{g in half h} Md[o,g] * sum_{j in g} c[o,j]*(g[j]-zg) ) + 128 )`, `h=0:inputs 0..255, h=1:inputs 256..511`,
`Md = fx(sw*s_g/s_dp)`, format `dp=(s_dp,128)`. Then `dn = clip8( rs( ((dp_0-128)+(dp_1-128)) * fx(s_dp/s_dn) ) + zdn )` (feComposite k2=k3), format `dn`.
`x_next = r2 = clip8( rs( (r1-zr1)*fx(s_r1/s_r2) + (dn-zdn)*fx(s_dn/s_r2) ) + zr2 )`.

### 1.9 Head
`nf = RMSNorm(x_L, gf)` (1.2). `logits[v] = clip8( rs( sum_g Mh[v,g]*sum_{j in g} c_emb[v,j]*(nf[j]-128) ) + zlg )`, 128 taps x 86 outputs, `Mh = fx(sw_emb*s_nf/s_lg)`.
Greedy token = **first** index of the maximum of the 86 uint8 logits (ties -> lowest index). EOS = 85 ends the answer.
Argmax in a filter graph: tournament of exact max ops (1.5 step 3) plus an equality indicator (LUT) against each logit.

## 2. Generation / prompt
`prompt = "q: " + q + "\na: "` where `q` = lowercased question, characters not in `VOC` dropped, whitespace runs collapsed to one space, stripped, cut to 140 chars
(identical to `qa_infer.answer`). Greedy decode, stop at EOS or 300 generated chars. `generate()` in `q8_ref.py` does this with a KV cache
(cached `kr` and `v` codes; exactly equivalent to full recompute because every op is position-wise/causal).

## 3. Sizes / context
* Max context seen over the 692 menu questions + the 6 presets with the fp model's answers: **215** tokens (prompt <= 78, answer <= 150).
  The model was trained with answers <= 150 chars and questions <= 140 chars: absolute worst case 6+140+150 = 296.
* Recommend a **max context of 224** for the SVG graph if only menu/preset questions are used (T <= 256 => one PV block, one softmax-sum kernel, exact 256-tap limits).
  Use 256 for headroom. Past 256 every row-sum and PV sum needs a second 256-tap piece.
* Per layer, per query row and head the attention needs `T` products x 32 (scores) and `T` products x 32 (values); per answer token the graph is
  O(T) deep in primitives, as discussed in REPORT.md.

## 4. Files
`q8_ref.py` reference + `Q8Ref.export_tables`, `q8_params.npz` weights + formats, `q8_tables.npz` derived LUTs/multipliers,
`q8_common.py`/`q8_train.py` (torch QAT model that mirrors the reference), `train_q8.py`, `eval_ref.py`, `qa_q8.pt` (torch checkpoint incl. spec).

## 5. Results (shipped `qa_q8.pt` / `q8_params.npz`)
Reference = fp 4-bit README model (`qa_infer.answer` on `/Users/jhome/Bobliuuu/README.md`), greedy, `q8_ref.Q8Ref.generate`.
| set | exact answer match |
|---|---|
| 692 menu questions (first phrasing) + 6 presets (698) | **698/698 = 100%** (char accuracy of the rest: n/a) |
| 350 second phrasings (not in the eval set) | 350/350 = 100% |
| same 698 with +-1 level uniform noise added to every convolve-type sum (`Q8Ref(noise=1)`) | 697/698 = 99.9% |
| same with +-2 levels noise | 673/698 = 96.4% |
| post-training quantisation only (no QAT, same graph) | 520/698 = 74.5% |
Weights unchanged in kind: 4-bit, group 32, fp16 scales (README-size). No 8-bit weights were needed.
Training: 6 epochs, AdamW lr 2e-4 one-cycle, loss 0.5*CE + 0.5*KL to the fp 4-bit teacher, STE through every round/LUT, static scales from calibration
(64 sequences, 99.99th percentile), 11.9k sequences (typo/prefix augmented menu data + menu questions with teacher answers).
The torch model uses float32 and is only *nearly* bit-identical to the integer reference (about 59% of last-position logit bytes equal on random prefixes, argmax 40/40); the
reference is the definition and the accuracy numbers above are from it.
