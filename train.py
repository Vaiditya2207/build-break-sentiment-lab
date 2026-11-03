"""Train Naive Bayes + Word2Vec on SST-2, save to artifacts/, print validation accuracy of all 4 models."""
import os

import joblib
import numpy as np
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from models import ART, W2VAvg, tokenize


def train_and_save():
    os.makedirs(ART, exist_ok=True)
    train = load_dataset("nyu-mll/glue", "sst2")["train"]
    texts, labels = train["sentence"], train["label"]

    vec = CountVectorizer()
    nb = MultinomialNB().fit(vec.fit_transform(texts), labels)
    joblib.dump((vec, nb), os.path.join(ART, "nb.joblib"))

    w2v = Word2Vec([tokenize(t) for t in texts], vector_size=100, min_count=1, epochs=10, seed=0, workers=1)
    w2v.save(os.path.join(ART, "w2v.model"))
    embed = W2VAvg(w2v.wv, None).embed
    X, y = [], []
    for t, l in zip(texts, labels):
        v = embed(t)
        if v is not None:
            X.append(v)
            y.append(l)
    clf = LogisticRegression(max_iter=1000).fit(np.array(X), y)
    joblib.dump(clf, os.path.join(ART, "w2v_clf.joblib"))


if __name__ == "__main__":
    train_and_save()
    from models import load_models
    val = load_dataset("nyu-mll/glue", "sst2")["validation"]
    gold = ["positive" if l == 1 else "negative" for l in val["label"]]
    for name, m in load_models().items():
        acc = np.mean([m.predict(t)[0] == g for t, g in zip(val["sentence"], gold)])
        print(f"{name:15s} SST-2 val accuracy: {acc:.3f}")
