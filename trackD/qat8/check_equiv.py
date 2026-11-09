"""torch QAT model vs integer reference: last-position logits equality on random prefixes."""
import sys, json, random, numpy as np, torch
sys.path.insert(0, "/Users/jhome/readme-lm/trackD/qat8")
from q8_train import *; from data8 import *
import q8_ref
D = "/Users/jhome/readme-lm/trackD/qat8/"
ck = torch.load(D + "qa_q8.pt", map_location="cpu"); m = Q8(); m.load_state_dict(ck["sd"]); m.spec = ck["spec"]; m = m.to(dev).eval()
r = q8_ref.Q8Ref(D + "q8_params.npz"); r.export_tables(D + "q8_tables.npz")
sp = ck["spec"]; print({l: (sp[f"L{l}.D"], sp[f"L{l}.f.z"], round(sp[f"L{l}.sc.s"], 3), sp[f"L{l}.sc.z"]) for l in range(3)})
T = json.load(open(D + "teacher_answers.json")); random.seed(1); eq = tot = full = 0; am = 0
for t in random.sample(T, 40):
    ids = enc(build_qa.norm_q(t["q"]), t["a"])[0]; ids = ids[:random.randint(20, len(ids))]
    lg = r.forward_token(ids)
    with torch.no_grad(): _, lc = m(torch.tensor([ids]).to(dev))
    tl = lc[0, -1].cpu().numpy().astype(np.uint8); eq += int((tl == lg).sum()); tot += len(lg); full += int((tl == lg).all()); am += int(tl.argmax() == lg.argmax())
print("torch-vs-ref logit element equality", eq / tot, "; all-86-equal", full, "/40; argmax equal", am, "/40")
