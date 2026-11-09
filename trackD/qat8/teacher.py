import sys, json, os
sys.path.insert(0, "/Users/jhome/readme-lm"); os.chdir("/Users/jhome/readme-lm")
import build_qa, qa_infer
from multiprocessing import Pool
PRESETS = ["what is qwentpu", "which repo runs doom inside qwen", "what is the ten percent challenge", "what language is allgit written in",
           "what is qweninqwen", "where does jerry study", "what is bobliu-os", "what is README-LM"]
def questions():
    items, _ = build_qa.load(); return [o["qs"][0] for o in items]
def _a(q):
    global M
    if "M" not in globals(): M = qa_infer.load("/Users/jhome/Bobliuuu/README.md")
    return qa_infer.answer(M, q)
if __name__ == "__main__":
    qs = questions() + PRESETS[:6]
    with Pool(10) as p: ans = p.map(_a, qs, chunksize=8)
    json.dump([{"q": q, "a": a} for q, a in zip(qs, ans)], open("/Users/jhome/readme-lm/trackD/qat8/teacher_answers.json", "w"))
    print(len(qs), sum(len(a) for a in ans)/len(ans), max(len(a) for a in ans))
