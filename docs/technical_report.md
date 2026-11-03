# Build-Break Sentiment Lab: Technical Report

## Abstract

We built a working tool that stress-tests sentiment models the way a build-break cycle stresses
software. Given any sentence, it generates five stress variants (negation, synonym swap, sarcasm,
typo, out-of-vocabulary word), runs four models from four NLP paradigms side by side (rule-based
lexicon, Naive Bayes, averaged Word2Vec, BERT), and labels each failure as Architectural,
Data-driven or Evaluation. On SST-2, accuracy rises from 0.563 (rule-based) to 0.911 (BERT), yet
every model collapses on negation (≤ 0.465) and sarcasm (≤ 0.410). BERT makes zero Architectural
errors; all of its failures are Data-driven or come from doubtful tests. A targeted fix for the
baselines (bigrams, a negation window) adds +1.4 points of accuracy but barely improves negation
robustness. That supports the view that negation is an architectural limit of bag-of-words models.

## 1. Problem and motivation

Benchmark accuracy hides *how* a model breaks. Teams need to know whether a failure needs a new
model family, more data, or a better test. Full brief: `01_problem_brief.md`.

## 2. Data

SST-2 (GLUE; Socher et al., 2013). 67,349 training phrases (55.8% positive) and 872 validation
sentences (50.9% positive). Test labels are hidden, so validation is our held-out set and nothing
is tuned on it. There is no train/validation overlap. 5.82% of validation tokens are out of the
Naive Bayes vocabulary. Training phrases are much shorter (median 6 tokens) than validation
sentences (median 17). Negation appears in 24.3% of negative and 13.5% of positive validation
sentences. Full data card: `02_data_card.md`.

## 3. Method

### 3.1 Models (one per paradigm)

| Model | Paradigm | Details |
|---|---|---|
| Rule-based | Symbolic | 48-word hand lexicon; majority vote; no negation handling |
| Naive Bayes | Statistical bag of words | unigram counts + MultinomialNB |
| Word2Vec (avg) | Distributional semantics | 100-d Word2Vec trained on SST-2, mean vector → logistic regression |
| BERT | Pretrained transformer | DistilBERT fine-tuned on SST-2 (Hugging Face) |

### 3.2 Stress variants (`perturb.py`, deterministic)

| Kind | Example ("The movie was great") | Expected label |
|---|---|---|
| Negation | The movie was **not** great | flipped |
| Synonym | The movie was **terrific** | same |
| Sarcasm | **Oh sure,** the movie was great**. Yeah, right.** | negative |
| Typo | The **mvoie** was great | same |
| OOV | The movie was **bussin** | same |

### 3.3 Failure labelling (`diagnose.py`)

- **Evaluation**: the test is doubtful (partial negation of a multi-clause sentence; sarcasm on an already negative sentence; every model fails a meaning-preserving edit).
- **Architectural**: an order-insensitive model meets negation or sarcasm.
- **Data-driven**: an unseen word, or a learned association that is weak or wrong.

## 4. Results

### 4.1 Accuracy (SST-2 validation, 872)

Rule-based 0.563 · Naive Bayes 0.806 · Word2Vec (avg) 0.712 · BERT 0.911

### 4.2 Robustness (first 200 validation sentences + variants)

| Model | original | negation | synonym | sarcasm | typo | oov |
|---|---|---|---|---|---|---|
| Rule-based | 0.535 | 0.465 | 0.708 | 0.040 | 0.530 | 0.525 |
| Naive Bayes | 0.790 | 0.220 | 0.792 | 0.355 | 0.780 | 0.790 |
| Word2Vec (avg) | 0.685 | 0.335 | 0.792 | 0.410 | 0.690 | 0.675 |
| BERT | 0.910 | 0.320 | 0.958 | 0.390 | 0.900 | 0.895 |

### 4.3 Failure categories

| Model | Architectural | Data-driven | Evaluation |
|---|---|---|---|
| Rule-based | 246 | 201 | 141 |
| Naive Bayes | 213 | 94 | 111 |
| Word2Vec (avg) | 203 | 146 | 97 |
| BERT | 0 | 239 | 79 |

### 4.4 Improvement iteration

| Model | full val | negation (200) |
|---|---|---|
| Naive Bayes → + bigrams | 0.806 → 0.820 | 0.220 → 0.245 |
| Rule-based → + negation window | 0.563 → 0.568 | 0.465 → 0.475 |

Details and the full table: `04_evaluation.md`, `results/improvement.txt`.

## 5. Discussion

1. **Each paradigm fixes the previous one's main failure, except negation and sarcasm.** Naive Bayes learns words the lexicon lacks. BERT adds word order and context and so removes all Architectural failures.
2. **Word2Vec (avg) lost to Naive Bayes** (0.712 vs 0.806). Embeddings learned from 67k short phrases are weak, and averaging blurs strong single-word cues.
3. **Negation is hard even for BERT** (0.320). Many inserted negations leave long sentences with mixed sentiment, so part of this is an evaluation problem, which our Evaluation category makes visible.
4. **Patching bag-of-words models doesn't fix architecture.** Bigrams and a negation window help slightly overall but not on negation scope.

## 6. Deployment

Streamlit app (`app.py`). Models load in 7.2 s, and a full request (6 inputs × 4 models) takes
0.11–0.18 s on Apple Silicon. BERT median latency is 20.2 ms, p95 39.9 ms. Failure handling
(truncation, empty input, missing artifacts) and the monitoring plan are in `05_deployment.md`.

## 7. Limitations and threats to validity

- Template-based variants sometimes read unnaturally, so their expected labels can be wrong.
- Failure categories come from transparent rules, not causal analysis. Agreement with human judgement wasn't measured.
- Sarcasm is always expected to be negative, which is a simplification.
- One dataset and domain (English movie reviews). The robustness numbers use 200 sentences, a small sample.
- The BERT checkpoint was fine-tuned on SST-2 itself, so its clean accuracy is optimistic compared with the models we trained.

## 8. Future work

Hand-check 50 failures to measure category accuracy; add human-written negation and sarcasm test
sets (e.g. CheckList-style); fine-tune BERT on negation and sarcasm examples and re-measure its
Data-driven failures.

## References

1. R. Socher et al., "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank," EMNLP 2013.
2. A. Wang et al., "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding," ICLR 2019.
3. T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781, 2013.
4. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL 2019.
5. V. Sanh et al., "DistilBERT, a distilled version of BERT," arXiv:1910.01108, 2019.
6. M. T. Ribeiro et al., "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList," ACL 2020.
