"""Small, dependency-free text helpers shared by retrieval, grounding, RFAE and contradiction checks."""
import re

STOPWORDS = frozenset(
    """a an the of and or to in on at by for with from as is are was were be been being this that these those it its
    any all such shall may must will would should can could not no nor but if then than so also into upon under over per
    via etc each other their there where which who whom whose has have had do does did he she they them his her we our
    you your i my me us about between within without against during before after above below up down out off only own
    same both few more most some very s t just""".split()
)

# Words that make a gap description longer without changing what is being asked for.
GENERIC = frozenset(
    """detail details document documents evidence information copy copies full proof require required need needed
    relevant specific regarding related respective applicable concerned particulars particular""".split()
)

_SYNONYMS = {"valuation": "value", "valued": "value", "worth": "value", "lac": "lakh", "lacs": "lakh", "lakhs": "lakh"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(word):
    """Very light suffix stripper. Not linguistically correct, only consistent: date/dates/dated all map to 'dat'."""
    word = _SYNONYMS.get(word, word)
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("sses", "ches", "shes", "xes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokens(text, stop=True, do_stem=True):
    out = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        if stop and raw in STOPWORDS:
            continue
        out.append(stem(raw) if do_stem else raw)
    return out


def keywords(text):
    """Content words of a short phrase: stemmed, no stopwords, no filler, no bare numbers."""
    return frozenset(t for t in tokens(text) if not t.isdigit() and t not in GENERIC and stem(t) not in GENERIC)


def normalize_space(text):
    return " ".join((text or "").split())
