# M3: Baseline, Candidates and Experiment Log

All runs use SST-2 (GLUE). Train = 67,349 sentences, evaluation = the 872-sentence validation
split (test labels are hidden). Nothing was tuned on validation. Code is version-controlled in git;
each experiment below maps to a script you can re-run.

## Models

| # | Model | Paradigm | Setup |
|---|---|---|---|
| B0 | Rule-based | Symbolic | 24 positive + 24 negative hand-picked words; count and compare; tie → positive |
| B1 | Naive Bayes | Statistical, bag of words | `CountVectorizer()` unigrams + `MultinomialNB` |
| C1 | Word2Vec (avg) | Distributional embeddings | gensim Word2Vec (100-d) trained on SST-2 train, mean of word vectors → LogisticRegression |
| C2 | BERT | Pretrained transformer | `distilbert-base-uncased-finetuned-sst-2-english`, no extra training |

B0 and B1 are the baselines; C1 and C2 are the candidate improvements.

## Log

| Date | Exp | Script | Change | Result (measured) |
|---|---|---|---|---|
| 2026-11-03 | E1 | `train.py` | Train B1, C1; evaluate all 4 on full validation | Rule 0.563, NB 0.806, W2V 0.712, BERT 0.911 |
| 2026-11-03 | E2 | `evaluate.py 200` | Stress test on 200 val sentences × 6 inputs | Negation accuracy ≤ 0.465 for every model; BERT 0.320 (see `results/stress_test.txt`) |
| 2026-11-03 | E3 | `diagnose.py` v1 | "All 4 models fail → Evaluation" | Too blunt: real sarcasm failures were hidden as Evaluation |
| 2026-11-03 | E4 | `diagnose.py` v2 | Evaluation only for doubtful tests (partial negation of a multi-clause sentence, sarcasm on a negative sentence, all fail a meaning-preserving edit) | BERT: 0 Architectural, 239 Data-driven, 79 Evaluation |
| 2026-11-03 | E5 | `perturb.py` | Sentences already containing "not" were becoming "not not" | Negation now removes the existing "not" instead |
| 2026-11-03 | E6 | `improve.py` | B0 + negation window (3 tokens); B1 + bigrams | NB 0.806 → 0.820 full val; negation 0.220 → 0.245. Rule-based 0.563 → 0.568 (see `results/improvement.txt`) |
| 2026-11-03 | E7 | `models.py` | BERT `truncation=True` | 3,000-word input no longer crashes (see `results/benchmark.txt`) |

## Observations

- Accuracy goes up with each paradigm, except that Word2Vec trained only on SST-2 (0.712) loses to Naive Bayes (0.806). 67k short sentences aren't enough to learn good embeddings, and averaging throws away word order *and* the strong single-word signals Naive Bayes keeps.
- BERT wins on every variant except negation and sarcasm, where all models are weak.
