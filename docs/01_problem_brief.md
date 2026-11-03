# M1: Problem Brief and Business Case

## Problem

Teams use sentiment models to read customer reviews, support tickets and social posts. A model
that scores 90% on a benchmark can still flip its answer on a simple "not", a typo, new slang or
sarcasm, and nobody notices until wrong numbers reach a dashboard. Accuracy alone also doesn't
say *why* a model fails, so teams can't tell whether to switch model, collect more data, or fix
the test itself.

**Build-Break Sentiment Lab** takes any sentence, generates five stress variants (negation, synonym
swap, sarcasm, typo, out-of-vocabulary word), runs four sentiment models from four NLP paradigms,
and labels every failure as **Architectural**, **Data-driven** or **Evaluation**.

## Users

| User | What they need |
|---|---|
| ML / data engineers choosing a sentiment model | See where each model type breaks before deploying it |
| Product or CX analysts using sentiment dashboards | Know which inputs (slang, sarcasm, negation) to distrust |
| NLP students and instructors | See the classic-to-modern trade-offs on their own sentences |

## Business objective

Cut the cost of shipping a fragile sentiment model by finding robustness failures *before*
deployment, and point each failure to the right fix:

- Architectural → change model family
- Data-driven → add training data
- Evaluation → fix the test or the label

## Success metrics

| Metric | Target | Measured (see `results/`) |
|---|---|---|
| Best model accuracy on SST-2 validation | ≥ 0.85 | BERT 0.911 |
| Stress variants generated per sentence | 5 types, deterministic | 4–5 (synonym skipped if no known word) |
| App latency for one sentence (4 models × 6 inputs) | < 2 s on a laptop | 0.11–0.18 s per full request (measured) |
| Every failure gets a category and a reason | 100% | 100% (rule-based, by construction) |
| Category agrees with a human reviewer | ≥ 80% on 50 hand-checked failures | **not measured yet** (future work) |

## Feasibility

- **Data**: SST-2 is public (GLUE), with 67k labelled training sentences. No collection needed.
- **Compute**: everything runs on a laptop CPU or Apple GPU. Naive Bayes and Word2Vec train in a few minutes, and BERT is pretrained.
- **Risk**: the failure categories come from rules, not causal proof. We state this openly and use the Evaluation category to flag doubtful tests.
- **Scope**: English, binary sentiment, movie-review domain. Other domains are out of scope.
