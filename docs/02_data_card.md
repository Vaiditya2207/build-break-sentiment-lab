# Data Card: SST-2 (Milestone M2)

All numbers below come from `.venv/bin/python eda.py` (full output in `results/eda.txt`; plots in
`results/class_balance.png` and `results/length_hist.png`).

## Source and license

- **Dataset:** SST-2, the binary version of the Stanford Sentiment Treebank, as distributed in the GLUE
  benchmark. Loaded with `load_dataset("nyu-mll/glue", "sst2")` from the Hugging Face Hub.
- **Citation:** Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and Potts, C. (2013).
  *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP 2013.
  GLUE: Wang et al. (2018), *GLUE: A Multi-Task Benchmark and Analysis Platform for NLU.*
- **License:** released by the Stanford NLP group for research use. GLUE redistributes it, and the Hugging
  Face card does not list a separate license. We use it for coursework only.

## What it contains

English snippets of movie reviews, each labelled `0` (negative) or `1` (positive). The original source is
reviews from rottentomatoes.com, collected by Pang and Lee (2005). Socher et al. parsed each sentence into a
tree, and annotators on Amazon Mechanical Turk rated every phrase on a sentiment slider. SST-2 drops the
neutral phrases and keeps positive versus negative. The train split contains phrases as well as full
sentences. Validation and test contain full sentences only.

## Splits

| Split | Size | Negative | Positive | Use in this project |
|---|---|---|---|---|
| train | 67,349 | 29,780 (44.2%) | 37,569 (55.8%) | trains Naive Bayes and Word2Vec |
| validation | 872 | 428 (49.1%) | 444 (50.9%) | **our held-out test set** |
| test | 1,821 | hidden (all `-1`) | hidden | not used |

The GLUE test labels are hidden, so the validation split is our test set. We do **no tuning on it**: the
models use library defaults, and the fixed BERT checkpoint was already fine-tuned elsewhere. We use the
official splits as they are and do not re-split.

## Exploratory analysis

- **Class balance:** train leans positive (55.8%). Validation is close to 50/50. A model that always
  says "positive" would score 50.9% on validation.
- **Length (tokens):** train mean 8.9, median 6, p95 24, max 49. Validation mean 17.7, median 17, p95 32,
  max 44. 28.4% of train items have 3 tokens or fewer, against 0.7% of validation sentences. Train is
  mostly short phrases, and validation is mostly full sentences (see `length_hist.png`).
- **Top words (train, small stopword list removed):** negative: *n't, not, too, like, bad, no, just, up,
  i, does, ...* Positive: *good, most, funny, story, well, best, not, like, life, comedy, ...* Negation
  words are the strongest negative signal, but *not* and *n't* also appear in many positive phrases.
- **Negation (not / no / never / n't):** present in 13.0% of negative and 4.9% of positive train items, and
  in 24.3% of negative and 13.5% of positive validation sentences. Negation is roughly twice as common in
  validation because its sentences are longer. This matches the negation failures we see in the stress
  test.
- **Duplicates and leakage:** 371 duplicate sentences within train (0.55%), after lowercasing and
  stripping whitespace. **0 of 872** validation sentences appear exactly in train, so there is no exact
  train/test leakage.
- **Vocabulary:** 13,816 types in train with our tokenizer, and 13,774 in the Naive Bayes
  `CountVectorizer`. 841 of 14,447 validation tokens (5.82%) are out of the Naive Bayes vocabulary.

## Cleaning applied

None of the data is removed or relabelled. Each model does only light normalisation:
- Rule-based and Word2Vec: lowercase, then the regex `[a-z0-9']+` (`models.tokenize`).
- Naive Bayes: scikit-learn `CountVectorizer` defaults, which lowercase and keep word tokens of 2 or more
  characters.
- BERT: its own pretrained WordPiece tokenizer.

We keep the 371 duplicates in train. They are 0.55% of the data and do not touch validation.

## Known biases and limitations

- **One domain:** movie reviews only. Language from other domains (products, social media, slang) is
  under-represented, which is why our OOV/slang probes break the trained models.
- **Short phrases in train:** many train items are fragments like "too bad" or "charming". Their labels are
  clean, but they do not look like the full sentences people type (28.4% of train items have 3 tokens or
  fewer).
- **English only.** The text is also early-2000s critic prose, so newer slang is missing.
- **Binary labels:** neutral and mixed sentiment were removed, so a mixed sentence still has to be forced
  into one label. This is part of why our "Evaluation" failure category exists.
- Crowd-sourced labels carry some annotator noise.

## Privacy considerations

- The data is public movie-review snippets written by professional critics. It has no user accounts, no
  personal data, and no identifiers beyond what appears in the published reviews.
- The Streamlit app runs locally. It does **not store** the sentences users type: each one is processed in
  memory for the current prediction and then discarded, with nothing logged or saved to disk.
