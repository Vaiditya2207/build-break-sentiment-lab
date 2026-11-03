"""Four sentiment models sharing one tiny interface: predict, in_vocab, order_aware."""
import os
import re

import joblib
import numpy as np
from gensim.models import Word2Vec

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
POS_WORDS = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "best", "love",
             "loved", "enjoy", "enjoyable", "fun", "funny", "beautiful", "brilliant", "perfect",
             "awesome", "nice", "charming", "delightful", "moving", "touching", "superb", "masterpiece"}
NEG_WORDS = {"bad", "terrible", "awful", "horrible", "worst", "boring", "dull", "hate", "hated",
             "poor", "stupid", "waste", "mess", "disappointing", "annoying", "ugly", "weak",
             "mediocre", "pointless", "lame", "bland", "tedious", "fails", "worse"}


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class RuleBased:
    order_aware = False

    def predict(self, text):
        toks = tokenize(text)
        pos = sum(t in POS_WORDS for t in toks)
        neg = sum(t in NEG_WORDS for t in toks)
        if pos == neg:
            return "positive", 0.5
        label = "positive" if pos > neg else "negative"
        return label, max(pos, neg) / (pos + neg)

    def in_vocab(self, word):
        return word.lower() in POS_WORDS or word.lower() in NEG_WORDS


class NaiveBayes:
    order_aware = False

    def __init__(self, vec, clf):
        self.vec, self.clf = vec, clf

    def predict(self, text):
        p = self.clf.predict_proba(self.vec.transform([text]))[0][1]
        p = float(p)
        return ("positive", p) if p >= 0.5 else ("negative", 1 - p)

    def in_vocab(self, word):
        return word.lower() in self.vec.vocabulary_


class W2VAvg:
    order_aware = False

    def __init__(self, wv, clf):
        self.wv, self.clf = wv, clf

    def embed(self, text):
        vecs = [self.wv[t] for t in tokenize(text) if t in self.wv]
        return np.mean(vecs, axis=0) if vecs else None

    def predict(self, text):
        v = self.embed(text)
        if v is None:
            return "positive", 0.5
        p = self.clf.predict_proba([v])[0][1]
        p = float(p)
        return ("positive", p) if p >= 0.5 else ("negative", 1 - p)

    def in_vocab(self, word):
        return word.lower() in self.wv


class Bert:
    order_aware = True

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def predict(self, text):
        out = self.pipe(text, truncation=True)[0]
        return out["label"].lower(), float(out["score"])

    def in_vocab(self, word):
        return True


def load_models():
    if not os.path.exists(os.path.join(ART, "w2v_clf.joblib")):
        import train
        train.train_and_save()
    vec, nb = joblib.load(os.path.join(ART, "nb.joblib"))
    wv = Word2Vec.load(os.path.join(ART, "w2v.model")).wv
    w2v_clf = joblib.load(os.path.join(ART, "w2v_clf.joblib"))
    return {
        "Rule-based": RuleBased(),
        "Naive Bayes": NaiveBayes(vec, nb),
        "Word2Vec (avg)": W2VAvg(wv, w2v_clf),
        "BERT": Bert(),
    }
