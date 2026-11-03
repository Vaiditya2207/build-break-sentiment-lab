"""M5 performance: model load time, size on disk, and per-sentence latency (median / p95).

Run:  .venv/bin/python benchmark.py   (writes results/benchmark.txt)
"""
import os
import platform
import time

import numpy as np
import torch
from datasets import load_dataset

from models import ART, load_models

t0 = time.perf_counter()
models = load_models()
load_s = time.perf_counter() - t0

sents = load_dataset("nyu-mll/glue", "sst2")["validation"]["sentence"][:200]
size = {"Rule-based": 0,
        "Naive Bayes": os.path.getsize(f"{ART}/nb.joblib"),
        "Word2Vec (avg)": sum(os.path.getsize(f"{ART}/{f}") for f in os.listdir(ART) if f.startswith("w2v")),
        "BERT": sum(p.numel() * p.element_size() for p in models["BERT"].pipe.model.parameters())}

lines = [f"Machine: {platform.machine()} / {platform.processor() or platform.platform()}, "
         f"torch {torch.__version__}, BERT device: {models['BERT'].pipe.device}",
         f"Load time for all 4 models: {load_s:.1f} s", "",
         f"{'Model':16} {'median ms':>10} {'p95 ms':>8} {'size MB':>8}"]
for name, m in models.items():
    for s in sents[:5]:
        m.predict(s)  # warmup
    times = []
    for s in sents:
        t = time.perf_counter()
        m.predict(s)
        times.append((time.perf_counter() - t) * 1000)
    lines.append(f"{name:16} {np.median(times):10.2f} {np.percentile(times, 95):8.2f} {size[name] / 1e6:8.1f}")

long_text = "great " * 3000
lines += ["", f"Failure handling: 3000-word input -> BERT {models['BERT'].predict(long_text)} (truncated to 512 tokens, no crash)"]
out = "\n".join(lines) + "\n"
print(out)
open("results/benchmark.txt", "w").write(out)
