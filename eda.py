"""Exploratory data analysis of SST-2 (handbook M2). Writes results/eda.txt and two PNGs.

Run:  .venv/bin/python eda.py
"""
import os
import re
from collections import Counter

import joblib
import matplotlib
import numpy as np
from datasets import load_dataset

from models import ART, tokenize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STOP = {"the", "a", "an", "and", "of", "to", "is", "it", "in", "that", "this", "as", "with", "for",
        "its", "it's", "on", "be", "by", "but", "his", "her", "are", "at", "from", "has",
        "have", "was", "who", "you", "or", "all", "more", "than", "so", "one", "about", "there",
        "their", "they", "s", "'s", "'", "if", "into", "out", "what", "film", "movie"}
NEG = re.compile(r"\b(not|no|never)\b|n't\b", re.I)
NAMES = {0: "negative", 1: "positive"}

os.makedirs(OUT, exist_ok=True)
ds = load_dataset("nyu-mll/glue", "sst2")
lines = []
say = lambda s="": lines.append(s)  # noqa: E731

say("SST-2 (GLUE) exploratory analysis")
say("Note: SST-2 test labels are hidden (all -1), so the validation split is our held-out test set;")
say("no tuning is done on it.\n")

say("== Split sizes and class balance ==")
lengths = {}
for split in ["train", "validation", "test"]:
    d = ds[split]
    counts = Counter(d["label"])
    bal = ", ".join(f"{NAMES.get(k, 'hidden(-1)')}={v} ({v / len(d):.1%})" for k, v in sorted(counts.items()))
    say(f"{split:10s} n={len(d):6d}  {bal}")
    lengths[split] = np.array([len(tokenize(s)) for s in d["sentence"]])

say("\n== Sentence length (tokens, models.tokenize) ==")
for split, L in lengths.items():
    say(f"{split:10s} mean={L.mean():.1f} median={np.median(L):.0f} p95={np.percentile(L, 95):.0f} max={L.max()}"
        f"  <=3 tokens={np.mean(L <= 3):.1%}")

train, val = ds["train"], ds["validation"]
say("\n== Top 20 words per class (train, stopwords removed) ==")
for lab in (0, 1):
    c = Counter(t for s, l in zip(train["sentence"], train["label"]) if l == lab
                for t in tokenize(s) if t not in STOP)
    say(f"{NAMES[lab]}: " + ", ".join(f"{w}({n})" for w, n in c.most_common(20)))

say("\n== % sentences with negation (not/no/never/n't) ==")
for split in ["train", "validation"]:
    d = ds[split]
    for lab in (0, 1):
        sents = [s for s, l in zip(d["sentence"], d["label"]) if l == lab]
        say(f"{split:10s} {NAMES[lab]:8s} {np.mean([bool(NEG.search(s)) for s in sents]):.1%}")

say("\n== Duplicates and leakage ==")
norm = lambda s: s.strip().lower()  # noqa: E731
tr = [norm(s) for s in train["sentence"]]
dups = len(tr) - len(set(tr))
say(f"duplicate sentences within train: {dups} ({dups / len(tr):.2%})")
tr_set = set(tr)
overlap = [s for s in (norm(v) for v in val["sentence"]) if s in tr_set]
say(f"validation sentences that also appear exactly in train: {len(overlap)} of {len(val)}")

say("\n== Vocabulary ==")
vocab = {t for s in train["sentence"] for t in tokenize(s)}
say(f"train vocabulary size (models.tokenize): {len(vocab)}")
vec, _ = joblib.load(os.path.join(ART, "nb.joblib"))
analyze = vec.build_analyzer()
val_toks = [t for s in val["sentence"] for t in analyze(s)]
oov = sum(t not in vec.vocabulary_ for t in val_toks)
say(f"Naive Bayes vocabulary size (CountVectorizer): {len(vec.vocabulary_)}")
say(f"validation tokens OOV w.r.t. NB vocabulary: {oov} of {len(val_toks)} ({oov / len(val_toks):.2%})")

text = "\n".join(lines)
print(text)
with open(os.path.join(OUT, "eda.txt"), "w") as f:
    f.write(text + "\n")

# Plot 1: class balance per labelled split
fig, ax = plt.subplots(figsize=(5, 3.5))
for i, split in enumerate(["train", "validation"]):
    c = Counter(ds[split]["label"])
    ax.bar([i - 0.2, i + 0.2], [c[0], c[1]], width=0.4, color=["tab:red", "tab:green"])
ax.set_xticks([0, 1], ["train", "validation"])
ax.set_yscale("log")
ax.set_ylabel("sentences (log scale)")
ax.set_title("SST-2 class balance (red=negative, green=positive)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "class_balance.png"), dpi=120)

# Plot 2: sentence length histogram
fig, ax = plt.subplots(figsize=(5, 3.5))
bins = np.arange(0, 60, 2)
for split in ["train", "validation"]:
    ax.hist(lengths[split], bins=bins, density=True, alpha=0.5, label=split)
ax.set_xlabel("tokens per sentence")
ax.set_ylabel("fraction of sentences")
ax.set_title("SST-2 sentence length")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "length_hist.png"), dpi=120)
