# M5: Deployment Prototype

## Working application

- **App**: Streamlit, `app.py`. Run with `.venv/bin/streamlit run app.py`, then open http://localhost:8501.
- **Flow**: type a sentence → pick its true label (defaults to BERT's guess) → see predictions for the original and 5 variants across 4 models → failure table with category and reason → counts per category and a per-model bar chart.
- **Batch mode**: `evaluate.py N` runs the same pipeline over N SST-2 sentences.
- **Model deployment**: Naive Bayes and Word2Vec are saved in `artifacts/` and loaded once (`st.cache_resource`). BERT comes from the Hugging Face hub and is cached locally after the first download. If the artifacts are missing, `load_models()` retrains them automatically.

## Performance (measured, `results/benchmark.txt`)

Apple Silicon (arm64), torch 2.15, BERT on the MPS GPU, 200 single-sentence calls after warmup:

| Model | median ms | p95 ms | size on disk |
|---|---|---|---|
| Rule-based | 0.01 | 0.01 | ~0 MB |
| Naive Bayes | 0.20 | 0.27 | 0.6 MB |
| Word2Vec (avg) | 0.11 | 0.17 | 11.5 MB |
| BERT | 20.17 | 39.86 | 267.8 MB |

- Loading all 4 models takes 7.2 s, once per app start.
- One full app request (6 inputs × 4 models, plus the default-label guess) measured 0.11–0.18 s, mostly BERT.

## Failure handling

| Risk | Handling |
|---|---|
| Empty input | App stops early (`st.stop()`) |
| Very long input (> 512 tokens) | BERT `truncation=True`; tested on a 3,000-word input, no crash |
| No known words for Word2Vec | Returns `positive, 0.5` instead of crashing on an empty average |
| Rule-based tie / no lexicon words | Returns `positive, 0.5` (shown as low confidence) |
| Missing model files | `load_models()` retrains Naive Bayes and Word2Vec automatically |
| No internet on first run | BERT/SST-2 download fails. Run `train.py` once with internet; everything is cached afterwards |

## Monitoring strategy (for a production version)

The prototype doesn't log anything. For a real deployment we would track:

1. **Latency**: p50/p95 per model; alert if BERT p95 > 200 ms.
2. **Input drift**: share of tokens that are OOV for the Naive Bayes vocabulary (5.82% on SST-2 validation). A rising OOV rate means new slang or a new domain, so collect data and retrain.
3. **Disagreement rate**: how often BERT and Naive Bayes disagree. A spike signals inputs the models were never trained on.
4. **Confidence**: share of BERT predictions below 0.6 confidence; send these for human review.
5. **Failure-category mix**: run `evaluate.py` weekly on a fresh labelled sample. More Data-driven failures → add training data. More Evaluation failures → fix the stress tests.

## Privacy

Typed sentences are processed in memory only. Nothing is stored, logged or sent anywhere except
the local models. Any production monitoring would log only aggregate counts, never raw text.
