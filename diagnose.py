"""Label a model failure as Architectural, Data-driven, or Evaluation.

Simple, explainable rules:
  Evaluation    - the test case itself is doubtful: its expected label is ambiguous, or every
                  model "fails" a change that was supposed to keep the meaning.
  Architectural - an order-insensitive model (bag of words / averaged vectors) meets a change
                  that only word order or context can capture (negation, sarcasm).
  Data-driven   - the model could in principle handle it, but its training data didn't cover
                  the word or pattern (OOV, typo, unseen synonym, weak association).
"""
import re

CLAUSE_BREAK = re.compile(r",|;|\b(and|but|although|though|while)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|nothing)\b|n't\b", re.I)


def diagnose(model_name, model, variant, all_failed, original_text, original_label):
    kind, word = variant["kind"], (variant.get("changed") or "").lower()

    # --- Evaluation: is the test (not the model) the problem? ---
    if kind == "negation" and word == "not" and CLAUSE_BREAK.search(original_text):
        return "Evaluation", "Only the first clause was negated, so the sentence now has mixed sentiment and the expected label is doubtful."
    if kind == "sarcasm" and original_label == "negative":
        return "Evaluation", "Sarcasm wrapped around an already negative sentence has no clear expected label."
    if all_failed and kind in ("synonym", "typo", "oov"):
        return "Evaluation", "Every model changed its answer, so this 'meaning-preserving' edit probably changed the meaning."

    # --- Architectural: the model's design can't represent the change ---
    if kind in ("negation", "sarcasm") and not model.order_aware:
        return "Architectural", f"{model_name} ignores word order, so it can't tell how '{word}' changes the meaning of the other words."
    if NEGATION.search(variant["text"]) and not model.order_aware:
        return "Architectural", f"The sentence contains a negation, and {model_name} can't link 'not' to the word it negates."

    # --- Data-driven: the training data didn't cover it ---
    unseen = [w for w in word.split() if not model.in_vocab(w)]
    if kind in ("typo", "oov", "synonym") and unseen:
        return "Data-driven", f"'{' '.join(unseen)}' never appeared in {model_name}'s training data or lexicon, so the model has no evidence about it."
    if kind == "sarcasm":
        return "Data-driven", f"{model_name} reads context, but SST-2 has almost no sarcastic examples to learn the cue from."
    return "Data-driven", f"{model_name} knows the words, but its training data gave it a weak or wrong association for this case."
