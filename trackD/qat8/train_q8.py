"""QAT fine-tune of the 8-bit-activation model. usage: python train_q8.py OUTTAG EPOCHS LR [resume.pt]"""
import sys, json, random, time, torch, numpy as np, torch.nn.functional as F
from data8 import *
from q8_train import *
from q8_export import export
from qa_model import QA
tag, EPOCHS, LR = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); resume = sys.argv[4] if len(sys.argv) > 4 else None
NOISE = float(__import__("os").environ.get("TNOISE", 0)); OUT = "/Users/jhome/readme-lm/trackD/qat8/"; BS = int(__import__("os").environ.get("BS", 24))
torch.manual_seed(0); random.seed(0)
sd = torch.load("/Users/jhome/readme-lm/qa_final2.pt", map_location="cpu")
teacher = QA(); teacher.load_state_dict(sd); teacher.set_qat(True); teacher = teacher.to(dev).eval()
train, _ = build_qa.make(holdout=0.0)
T = json.load(open(OUT + "teacher_answers.json"))
train = [(q, a) for q, a in train] + [(build_qa.norm_q(t["q"]), t["a"]) for t in T] * 2   # menu Qs with the fp model's own answers
data = [enc(q, a) for q, a in train]; print("train seqs", len(data), "max len", max(len(d[0]) for d in data), flush=True)
m = Q8.from_fp(sd).to(dev)
if resume:
    ck = torch.load(resume, map_location="cpu"); m.load_state_dict(ck["sd"]); m.spec = ck["spec"]; print("resumed", resume, flush=True)
else:
    X, l = batch([d[0] for d in random.sample(data, 64)])
    with torch.no_grad(): m(X, l)
    print("calibrated", len(m.spec), flush=True)
evalseq = [enc(build_qa.norm_q(t["q"]), t["a"]) for t in T[::3]]
def tf_eval():
    a = tot = 0
    with torch.no_grad():
        for i in range(0, len(evalseq), 16):
            ch = evalseq[i:i + 16]; X, l = batch([c[0] for c in ch]); lg, _ = m(X, l); ar = torch.arange(X.shape[1], device=dev)[None]
            mk = (ar < l[:, None] - 1) & (ar >= torch.tensor([c[1] for c in ch], device=dev)[:, None] - 1)
            a += int(((lg.argmax(-1) == X.roll(-1, 1)) & mk).sum()); tot += int(mk.sum())
    return a / tot
opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=0.0, betas=(0.9, 0.98))
nb = len(data) // BS; sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb + 5, pct_start=0.1)
print("tf-agree before", tf_eval(), flush=True)
for ep in range(EPOCHS):
    t0 = time.time(); idx = sorted(range(len(data)), key=lambda i: len(data[i][0]) + random.random() * 20)
    batches = [idx[i:i + BS] for i in range(0, len(idx) - BS + 1, BS)]; random.shuffle(batches); tl = 0; m.train(); m.noise = NOISE
    for bi, b in enumerate(batches):
        X, l = batch([data[i][0] for i in b]); pl = torch.tensor([data[i][1] for i in b], device=dev)
        ar = torch.arange(X.shape[1] - 1, device=dev)[None]; mk = ((ar < l[:, None] - 1) & (ar >= pl[:, None] - 1)).float()
        lg, _ = m(X[:, :-1], torch.clamp(l - 1, min=1))
        with torch.no_grad(): tlg = teacher(X[:, :-1])
        ce = (F.cross_entropy(lg.reshape(-1, V), X[:, 1:].reshape(-1), reduction="none") * mk.reshape(-1)).sum() / mk.sum()
        kl = (F.kl_div(F.log_softmax(lg, -1), F.softmax(tlg, -1), reduction="none").sum(-1) * mk).sum() / mk.sum()
        loss = 0.5 * ce + 0.5 * kl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); tl += loss.item()
        if bi % 50 == 0: print(f"  ep{ep} b{bi}/{len(batches)} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    m.noise = 0.0; m.eval(); torch.save({"sd": m.state_dict(), "spec": m.spec, "ep": ep}, OUT + f"{tag}.pt"); export(m, OUT + f"{tag}_params.npz")
    print(f"ep {ep} loss {tl/len(batches):.4f} tf-agree {tf_eval():.4f} {time.time()-t0:.0f}s", flush=True)
