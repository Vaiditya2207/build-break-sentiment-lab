# Viva Prep: Likely Questions and Short Answers

**Why these four models?**
One per paradigm in the course: symbolic (rules), statistical (Naive Bayes), distributional (Word2Vec), pretrained contextual (BERT). Each one fixes a weakness of the previous one.

**Why does Word2Vec do worse than Naive Bayes?**
We trained it on only 67k short SST-2 phrases, so the vectors are weak. Averaging also blurs strong single words like "worst". Naive Bayes keeps every word's own weight.

**How do you decide Architectural vs Data-driven?**
Architectural: the model *can't* represent the change, e.g. a bag-of-words model can't see that "not" applies to "great". Data-driven: it could, but training didn't cover it, e.g. it never saw "bussin". BERT reads word order, so it never gets Architectural.

**What is an Evaluation failure?**
The test is wrong, not the model. Example: "The movie was not great and the acting was good" is mixed, so labelling it negative is doubtful.

**Is your category labelling correct?**
It's rule-based and transparent, not causal proof. We haven't measured agreement with human labels yet. That's our main limitation and next step.

**Why is BERT bad at negation (0.32)?**
Partly real (SST-2 has few clear negation examples), and partly our test: one "not" in a long multi-clause sentence often doesn't flip the whole sentence.

**What did the improvement iteration show?**
Bigrams gave Naive Bayes +1.4 points overall (0.806 → 0.820) but only +2.5 on negation. Local patches don't fix negation scope, which confirms it's architectural.

**Isn't BERT's 0.911 unfair?**
Yes, a bit: that checkpoint was fine-tuned on SST-2 training data. Our other models were also trained on SST-2 train, and validation is never used for training.

**Train/val/test split?**
SST-2 test labels are hidden, so validation (872) is our held-out test set. No tuning on it, and no overlap with train.

**How fast is it?**
BERT median 20 ms per sentence on Apple Silicon. A full app request takes 0.11–0.18 s.

**How would you monitor it in production?**
Track the OOV rate (drift), BERT vs Naive Bayes disagreement, the low-confidence share, latency, and the weekly failure-category mix (`05_deployment.md`).

**Privacy?**
Public movie-review data. Typed sentences are never stored or logged.
