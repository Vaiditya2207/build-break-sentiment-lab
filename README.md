# Build-Break Sentiment Lab

Type a sentence. The app makes five stress variants of it (negation, synonym swap, sarcasm, typo,
out-of-vocabulary slang) and runs four sentiment models side by side, one per paradigm from the course:

| Model | Paradigm | Trained on |
|---|---|---|
| Rule-based | hand-written lexicon, count positive vs negative words | nothing |
| Naive Bayes | bag-of-words + probabilistic classifier | SST-2 train |
| Word2Vec (avg) | averaged word embeddings + logistic regression | SST-2 train |
| BERT | pretrained transformer (DistilBERT fine-tuned on SST-2) | pretrained |

Every wrong prediction is labelled with one of three categories:

- **Architectural**: the model's design can't represent the change. Bag-of-words and averaged-vector models ignore word order, so negation and sarcasm break them.
- **Data-driven**: the design could handle it, but the training data didn't cover the word or pattern (slang, typos, rare synonyms, sarcasm for BERT).
- **Evaluation**: the test itself is suspect. Its expected label is ambiguous (e.g. only one clause negated), or every model "fails" an edit that was supposed to keep the meaning.

## Run

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python train.py            # trains NB + Word2Vec, prints SST-2 val accuracy (~5 min first time)
.venv/bin/streamlit run app.py       # the demo
.venv/bin/python evaluate.py 200     # batch stress test on 200 SST-2 validation sentences
```

## Files

- `models.py`: the 4 models, each with `predict(text) -> (label, confidence)`
- `train.py`: trains and saves Naive Bayes and Word2Vec to `artifacts/`
- `perturb.py`: deterministic stress variants (`.venv/bin/python perturb.py` shows examples)
- `diagnose.py`: rules that label each failure Architectural / Data-driven / Evaluation
- `app.py`: Streamlit UI
- `evaluate.py`: batch robustness numbers for the report
- `improve.py`: improvement iteration (Naive Bayes + bigrams, rule-based + negation window)
- `benchmark.py`: latency, load time and model sizes
- `eda.py`: dataset analysis for the data card
- `results/`: saved outputs of the scripts above

## Submission documents (`docs/`)

| Milestone | Document |
|---|---|
| M1 Problem definition | `docs/01_problem_brief.md` |
| M2 Data | `docs/02_data_card.md`, `results/eda.txt`, `results/*.png` |
| M3 Models + experiment log | `docs/03_experiment_log.md` and git history |
| M4 Evaluation + improvement | `docs/04_evaluation.md`, `results/stress_test.txt`, `results/improvement.txt` |
| M5 Deployment | `app.py`, `docs/05_deployment.md`, `results/benchmark.txt` |
| M6 Final | `docs/technical_report.md`, `docs/slides.pptx`, `docs/viva_prep.md`, live demo |

## Results (measured)

SST-2 validation accuracy (872 sentences): Rule-based 0.563 · Naive Bayes 0.806 · Word2Vec (avg) 0.712 · BERT 0.911

Stress test on the first 200 SST-2 validation sentences (accuracy per variant):

| | original | negation | synonym | sarcasm | typo | oov |
|---|---|---|---|---|---|---|
| Rule-based | 0.535 | 0.465 | 0.708 | 0.040 | 0.530 | 0.525 |
| Naive Bayes | 0.790 | 0.220 | 0.792 | 0.355 | 0.780 | 0.790 |
| Word2Vec (avg) | 0.685 | 0.335 | 0.792 | 0.410 | 0.690 | 0.675 |
| BERT | 0.910 | 0.320 | 0.958 | 0.390 | 0.900 | 0.895 |

Failure categories (same run):

| | Architectural | Data-driven | Evaluation |
|---|---|---|---|
| Rule-based | 246 | 201 | 141 |
| Naive Bayes | 213 | 94 | 111 |
| Word2Vec (avg) | 203 | 146 | 97 |
| BERT | 0 | 239 | 79 |

Takeaways: BERT has no Architectural failures, because it reads word order. All of its failures are
Data-driven or Evaluation. Negation looks low for every model, BERT included, partly because SST-2
sentences are long and multi-clause: inserting one "not" often leaves mixed sentiment, which is an
evaluation problem rather than a model problem. That's why the Evaluation category exists.

## Limitations

- The perturbations are simple templates, not human-written. Some variants read oddly, so their expected label can be wrong.
- The failure categories come from transparent rules (model type + kind of edit + vocabulary check), not causal proof.
- Sarcasm always expects "negative", which is a simplification.
- The rule-based model returns "positive" on ties, so a few of its "correct" answers are luck.
