# M4: Evaluation, Error Analysis, Improvement and Robustness

## 1. Evaluation report

SST-2 validation (872 sentences, balanced 49.1% / 50.9%):

| Model | Accuracy |
|---|---|
| Rule-based | 0.563 |
| Naive Bayes | 0.806 |
| Word2Vec (avg) | 0.712 |
| BERT | 0.911 |

## 2. Robustness testing

Each of the first 200 validation sentences gets 5 stress variants (`perturb.py`, deterministic).
Expected labels: negation flips, sarcasm → negative, the others keep the original label.

| Model | original | negation | synonym | sarcasm | typo | oov |
|---|---|---|---|---|---|---|
| Rule-based | 0.535 | 0.465 | 0.708 | 0.040 | 0.530 | 0.525 |
| Naive Bayes | 0.790 | 0.220 | 0.792 | 0.355 | 0.780 | 0.790 |
| Word2Vec (avg) | 0.685 | 0.335 | 0.792 | 0.410 | 0.690 | 0.675 |
| BERT | 0.910 | 0.320 | 0.958 | 0.390 | 0.900 | 0.895 |

- Typos and OOV slang barely hurt (≤ 0.015 drop for BERT). Most sentences keep enough other sentiment words.
- Negation and sarcasm break every model.

## 3. Error analysis: three failure categories

| Model | Architectural | Data-driven | Evaluation |
|---|---|---|---|
| Rule-based | 246 | 201 | 141 |
| Naive Bayes | 213 | 94 | 111 |
| Word2Vec (avg) | 203 | 146 | 97 |
| BERT | 0 | 239 | 79 |

Rules (`diagnose.py`), checked in this order:

1. **Evaluation**: the test is doubtful. Examples: only the first clause was negated ("The movie was not great *and* the acting was good"), sarcasm was wrapped around an already negative sentence, or every model failed an edit that was meant to keep the meaning.
2. **Architectural**: an order-insensitive model (rule-based, Naive Bayes, averaged Word2Vec) meets negation or sarcasm, or any sentence containing a negation word.
3. **Data-driven**: the changed word isn't in the model's vocabulary or lexicon, or the model knows the words but learned the wrong association.

Representative cases (from the app):

| Input | Model | Prediction | Category | Why |
|---|---|---|---|---|
| "The movie was not great" | Rule-based | positive | Architectural | Counts "great", can't see "not" |
| "The movie was bussin" | Naive Bayes | negative | Data-driven | "bussin" never seen in training |
| "I despised the ending" | Rule-based | positive (tie) | Data-driven | "despised" missing from lexicon |
| "Oh sure, the movie was great. Yeah, right." | BERT | positive | Data-driven | SST-2 has almost no sarcasm to learn from |
| "The movie was not great and the acting was good" | all 4 | positive | Evaluation | Mixed sentiment; the "negative" label is doubtful |

Key finding: BERT has **zero Architectural failures**. Its remaining errors come from data (sarcasm,
negation patterns rare in SST-2) or from imperfect tests.

## 4. Improvement iteration

Hypothesis: the largest Architectural failure is negation, so give the bag-of-words baselines a way to see it.

| Model | full val (872) | negation | synonym | sarcasm | original (200) |
|---|---|---|---|---|---|
| Naive Bayes | 0.806 | 0.220 | 0.792 | 0.355 | 0.790 |
| **Naive Bayes + bigrams** | **0.820** | 0.245 | 0.833 | 0.390 | 0.760 |
| Rule-based | 0.563 | 0.465 | 0.708 | 0.040 | 0.535 |
| **Rule-based + negation window** | **0.568** | 0.475 | 0.750 | 0.050 | 0.545 |

Result: small, real gains (+1.4 points for Naive Bayes on full validation), but negation robustness
barely moved. Local patches ("not good" as a bigram, flipping the next 3 words) can't capture
negation scope in long multi-clause sentences, which supports labelling these failures
*Architectural*. Next steps would be a model that reads structure (BERT), or targeted negation
training data for BERT. Full output: `results/improvement.txt`.
