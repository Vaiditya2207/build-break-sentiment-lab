"""Build-Break demo: one sentence, five stress variants, four NLP paradigms.

Run:  .venv/bin/streamlit run app.py
"""
from collections import Counter

import pandas as pd
import streamlit as st

from diagnose import diagnose
from models import load_models
from perturb import make_variants

st.set_page_config(page_title="Build-Break Sentiment Lab", layout="wide")
st.title("Build-Break Sentiment Lab")
st.caption("Rule-based → Naive Bayes → averaged Word2Vec → BERT, stress-tested on one sentence.")

models = st.cache_resource(load_models)()

text = st.text_input("Enter a sentence", "The movie was great")
if not text.strip():
    st.stop()

bert_guess = models["BERT"].predict(text)[0]
label = st.radio("True sentiment of your sentence", ["positive", "negative"],
                 index=0 if bert_guess == "positive" else 1, horizontal=True)

cases = [{"kind": "original", "text": text, "expected": label, "changed": ""}] + make_variants(text, label)

# Run every model on every case.
rows, failures = [], []
for case in cases:
    preds = {name: m.predict(case["text"]) for name, m in models.items()}
    all_failed = all(p[0] != case["expected"] for p in preds.values())
    row = {"Test": case["kind"], "Sentence": case["text"], "Expected": case["expected"]}
    for name, (pred, conf) in preds.items():
        ok = pred == case["expected"]
        row[name] = f"{'✅' if ok else '❌'} {pred} ({conf:.2f})"
        if not ok:
            cat, why = diagnose(name, models[name], case, all_failed, text, label)
            failures.append({"Test": case["kind"], "Model": name, "Category": cat, "Why": why})
    rows.append(row)

st.subheader("Predictions")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("Failures")
if not failures:
    st.success("No failures. Try a harder sentence!")
else:
    fail_df = pd.DataFrame(failures)
    counts = Counter(f["Category"] for f in failures)
    cols = st.columns(3)
    for col, cat in zip(cols, ["Architectural", "Data-driven", "Evaluation"]):
        col.metric(cat, counts.get(cat, 0))
    st.dataframe(fail_df, hide_index=True, use_container_width=True)

    st.subheader("Failures per model")
    st.bar_chart(fail_df.groupby(["Model", "Category"]).size().unstack(fill_value=0))

with st.expander("What do the categories mean?"):
    st.markdown(
        "- **Architectural**: the model's design can't represent the change. Bag-of-words models ignore word order, so they miss negation and sarcasm.\n"
        "- **Data-driven**: the design could handle it, but the training data didn't cover the word or pattern (OOV words, typos, rare synonyms, sarcasm for BERT).\n"
        "- **Evaluation**: the test itself is suspect. Every model failed it, or its expected label is ambiguous."
    )
