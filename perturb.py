"""Deterministic text perturbations for probing sentiment models (see CONTRACT.md)."""
import re

AUX = {"is", "was", "are", "were", "am", "do", "does", "did", "can", "could",
       "will", "would", "should", "shall", "may", "might", "must", "has", "have", "had"}
SUBJECTS = {"i", "we", "you", "he", "she", "they", "it", "this", "that"}
SYNONYMS = {"good": "decent", "great": "terrific", "bad": "awful", "terrible": "dreadful",
            "boring": "dull", "hated": "despised", "loved": "adored", "love": "adore",
            "hate": "despise", "amazing": "wonderful", "awful": "horrible", "solid": "sturdy",
            "excellent": "superb", "funny": "hilarious", "beautiful": "lovely", "poor": "weak"}
SLANG = {"great": "bussin", "bad": "mid", "good": "gucci", "terrible": "trash-tier",
         "awful": "trash", "amazing": "fire", "excellent": "goated", "boring": "dusty",
         "loved": "stanned", "hated": "clowned on", "solid": "lowkey fire", "funny": "hilar"}


def flip(label):
    return "negative" if label == "positive" else "positive"


def words(text):
    return list(re.finditer(r"[A-Za-z'-]+", text))


def match_case(src, new):
    return new[0].upper() + new[1:] if src[0].isupper() else new


def lower_first(text):
    return text if re.match(r"I\b", text) else text[:1].lower() + text[1:]


def swap_first(text, table):
    """Replace the first word (in text order) found in `table`. Returns (new_text, new_word) or None."""
    for m in words(text):
        w = m.group().lower()
        if w in table:
            new = match_case(m.group(), table[w])
            return text[:m.start()] + new + text[m.end():], new
    return None


def negate(text):
    # Already negated? Remove the negation instead of stacking "not not".
    m = re.search(r"\s+not\b|n't\b", text, re.I)
    if m:
        return text[:m.start()] + text[m.end():], "not"
    for m in words(text):
        if m.group().lower() in AUX:
            return text[:m.end()] + " not" + text[m.end():], "not"
    # No auxiliary: full clause ("I hated it") vs. fragment ("A good film").
    has_subject = any(m.group().lower() in SUBJECTS for m in words(text))
    prefix = "It is not true that " if has_subject else "It is not true that this is "
    return prefix + lower_first(text), prefix.strip()


def typo(text):
    ws = sorted(words(text), key=lambda m: -len(m.group()))  # stable: first longest wins
    for m in ws:
        w = m.group()
        for i in range(1, len(w) - 2):
            if w[i] != w[i + 1]:
                new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
                return text[:m.start()] + new + text[m.end():], new
    m = ws[0]
    return text[:m.end()] + m.group()[-1] + text[m.end():], m.group() + m.group()[-1]


def make_variants(text, label):
    out = []
    t, c = negate(text)
    out.append({"kind": "negation", "text": t, "expected": flip(label), "changed": c})
    syn = swap_first(text, SYNONYMS)
    if syn:
        out.append({"kind": "synonym", "text": syn[0], "expected": label, "changed": syn[1]})
    core = lower_first(text.rstrip(" .!?"))
    out.append({"kind": "sarcasm", "text": f"Oh sure, {core}. Yeah, right.",
                "expected": "negative", "changed": "Oh sure, ... Yeah, right."})
    t, c = typo(text)
    out.append({"kind": "typo", "text": t, "expected": label, "changed": c})
    oov = swap_first(text, SLANG) or (text + " fr fr", "fr fr")
    out.append({"kind": "oov", "text": oov[0], "expected": label, "changed": oov[1]})
    return out


if __name__ == "__main__":
    examples = [("The movie was great", "positive"), ("I hated the ending", "negative"),
                ("A good, solid film", "positive"), ("Terrible acting and a boring plot", "negative")]
    for text, label in examples:
        print(f"\n{text!r} [{label}]")
        for v in make_variants(text, label):
            assert v["text"] != text, v
            print(f"  {v['kind']:9} -> {v['text']!r}  expected={v['expected']}  changed={v['changed']!r}")
        assert make_variants(text, label) == make_variants(text, label)  # deterministic
