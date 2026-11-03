"""Batch build-break run: stress-test all models on N SST-2 validation sentences.

Run:  .venv/bin/python evaluate.py [N]      (default N=200)
Prints accuracy per model per test type, and failure counts per category.
"""
import sys
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset

from diagnose import diagnose
from models import load_models
from perturb import make_variants

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
models = load_models()
data = load_dataset("nyu-mll/glue", "sst2", split="validation").select(range(n))

correct, total, cats = defaultdict(int), defaultdict(int), Counter()
for ex in data:
    label = "positive" if ex["label"] == 1 else "negative"
    cases = [{"kind": "original", "text": ex["sentence"], "expected": label, "changed": ""}]
    cases += make_variants(ex["sentence"], label)
    for case in cases:
        preds = {name: m.predict(case["text"])[0] for name, m in models.items()}
        all_failed = all(p != case["expected"] for p in preds.values())
        for name, pred in preds.items():
            total[(name, case["kind"])] += 1
            if pred == case["expected"]:
                correct[(name, case["kind"])] += 1
            else:
                cats[(name, diagnose(name, models[name], case, all_failed, ex["sentence"], label)[0])] += 1

acc = pd.Series({k: correct[k] / total[k] for k in total}).unstack()
print(f"\nAccuracy on {n} SST-2 validation sentences and their stress variants:\n")
print(acc.round(3).to_string())
print("\nFailure categories:\n")
print(pd.Series(cats).unstack(fill_value=0).to_string())
