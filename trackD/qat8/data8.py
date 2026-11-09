import sys, os, random, json, torch
sys.path.insert(0, "/Users/jhome/readme-lm"); os.chdir("/Users/jhome/readme-lm")
import build_qa
from q8_common import *
I = {c: i for i, c in enumerate(VOC)}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
def enc(q, a):
    p = [I[c] for c in "q: " + q + "\na: "]; r = [I[c] for c in a] + [EOS]
    return p + r, len(p)
def batch(seqs):
    L = max(len(x) for x in seqs); X = torch.full((len(seqs), L), EOS); M = torch.zeros(len(seqs), L)
    for j, x in enumerate(seqs): X[j, :len(x)] = torch.tensor(x); M[j, :len(x)] = 1
    return X.to(dev), torch.tensor([len(x) for x in seqs]).to(dev)
