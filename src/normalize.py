"""Independent text views; raw fields and address numbers remain available."""

import re
import unicodedata

SUFFIXES = frozenset("inc incorporated corp corporation llc ltd limited pvt private sarl sas sa eurl".split())
STREETS = {"st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard", "ln": "lane"}
ASCII_SEPARATORS = re.compile(r"[^a-z0-9]+")


def conservative(text):
    text = unicodedata.normalize("NFKC", text or "").casefold().replace("&", " and ")
    if text.isascii():
        return ASCII_SEPARATORS.sub(" ", text).strip()
    # Keep combining marks: dropping them corrupts names in Indic scripts.
    text = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text)
    return " ".join(text.split())


def accent_fold(text):
    # Fold accents on Latin letters only; retain Indic vowel signs and other scripts.
    if text.isascii():
        return text
    output, latin = [], False
    for c in unicodedata.normalize("NFD", text):
        mark=unicodedata.category(c).startswith("M")
        if not mark:
            latin = "LATIN" in unicodedata.name(c, "")
        if not (latin and mark):
            output.append(c)
    return unicodedata.normalize("NFC", "".join(output))


def name_views(text):
    normalized = conservative(text)
    tokens, suffix = normalized.split(), []
    while len(tokens) > 1 and tokens[-1] in SUFFIXES:
        suffix.insert(0, tokens.pop())
    return {"raw": text, "normalized": normalized, "accent_folded": accent_fold(normalized),
            "core": " ".join(tokens), "suffix": " ".join(suffix)}


def address_views(text):
    normalized = conservative(text)
    return {"raw": text, "normalized": normalized, "accent_folded": accent_fold(normalized),
            "standardized": " ".join(STREETS.get(t, t) for t in normalized.split()),
            "numbers": re.findall(r"\d+", normalized)}
