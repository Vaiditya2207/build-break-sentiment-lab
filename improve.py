"""M4 improvement iteration: fix the biggest Architectural failure (negation) in the two weakest
bag-of-words models, then re-run the same stress test.

  Rule-based + negation : flip a lexicon word's polarity if not/no/never/n't is in the 3 tokens before it
  Naive Bayes + bigrams : CountVectorizer(ngram_range=(1, 2)) so "not good" becomes its own feature

Run:  .venv/bin/python improve.py   (writes results/improvement.txt)
"""
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from models import NEG_WORDS, POS_WORDS, NaiveBayes, RuleBased, load_models, tokenize
from perturb import make_variants

NEGATORS = {"not", "no", "never", "n't", "dont", "don't", "isn't", "wasn't", "didn't", "doesn't"}


class RuleBasedNeg(RuleBased):
    def predict(self, text):
        toks, score = tokenize(text), 0
        for i, t in enumerate(toks):
            s = (t in POS_WORDS) - (t in NEG_WORDS)
            if s and (NEGATORS & set(toks[max(0, i - 3):i]) or any(w.endswith("n't") for w in toks[max(0, i - 3):i])):
                s = -s
            score += s
        if score == 0:
            return "positive", 0.5
        return ("positive" if score > 0 else "negative"), min(1.0, 0.5 + abs(score) / 4)


data = load_dataset("nyu-mll/glue", "sst2")
vec = CountVectorizer(ngram_range=(1, 2))
nb2 = MultinomialNB().fit(vec.fit_transform(data["train"]["sentence"]), data["train"]["label"])

base = load_models()
models = {"Rule-based": base["Rule-based"], "Rule-based + negation": RuleBasedNeg(),
          "Naive Bayes": base["Naive Bayes"], "Naive Bayes + bigrams": NaiveBayes(vec, nb2)}

val = data["validation"]
gold = ["positive" if y == 1 else "negative" for y in val["label"]]
full = {n: sum(m.predict(t)[0] == g for t, g in zip(val["sentence"], gold)) / len(gold) for n, m in models.items()}

correct, total = defaultdict(int), defaultdict(int)
for text, label in list(zip(val["sentence"], gold))[:200]:
    for case in [{"kind": "original", "text": text, "expected": label}] + make_variants(text, label):
        for n, m in models.items():
            total[(n, case["kind"])] += 1
            correct[(n, case["kind"])] += m.predict(case["text"])[0] == case["expected"]

table = pd.Series({k: correct[k] / total[k] for k in total}).unstack()
table.insert(0, "full_val_872", pd.Series(full))
out = ("Improvement iteration: before vs after (accuracy)\n"
       "full_val_872 = all 872 SST-2 validation sentences; other columns = first 200 + their stress variants\n\n"
       + table.round(3).to_string() + "\n")
print(out)
open("results/improvement.txt", "w").write(out)
