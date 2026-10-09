"""RFAE (request for additional evidence): rule-based gap detection, and merging of rule/NER/LLM gap lists.

Duplicate detection compares the *keywords* of two gap descriptions (stemmed content words, filler removed), so
'Date of breach and notice' and 'Dates of agreement, breach, cause of action and any notice' are recognised as the
same gap. An optional embedding check can confirm borderline lexical matches but never merges unrelated items."""
import re

from textutil import keywords

SEVERITY_RANK = {"critical": 0, "important": 1, "minor": 2}

# Tunable thresholds (see tests/test_rfae.py for the cases they were set against).
JACCARD_MATCH = 0.50      # |A&B| / |A|B|  - items share at least half of their combined keywords
OVERLAP_MATCH = 0.80      # |A&B| / min(|A|,|B|) - the shorter item is almost entirely contained in the longer one
OVERLAP_MIN_WORDS = 2     # ...but only trust containment when the shorter item has at least this many keywords
EMBED_MATCH = 0.85        # cosine for the embedding tie-break
EMBED_LEXICAL_FLOOR = 0.25  # embeddings may only confirm items that already share some wording


def rule_based_rfae(draft):
    text = draft.lower()
    items = []

    def add(item, why, severity):
        items.append({"item": item, "why": why, "severity": severity, "source": "rule"})

    if not re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b|\b(19|20)\d{2}\b", text):
        add("Dates of agreement, breach, cause of action and any notice", "Needed to compute limitation under the Limitation Act, 1963 and to test delay and laches.", "critical")
    if not re.search(r"valuation|value of the suit|suit value|₹|\brs\.?\s*\d|\binr\b|\blakh|\bcrore", text):
        add("Valuation of the suit or subject matter and amounts claimed", "Decides pecuniary jurisdiction, court fee under CPC and stamp duty.", "critical")
    if not re.search(r"stamp", text):
        add("Stamping details: stamp duty paid, denomination, state and date", "Unstamped instruments are inadmissible until duty and penalty are paid under the Indian Stamp Act, 1899.", "critical")
    if not re.search(r"cause of action|jurisdiction|situated|resides|registered office|carries on business|place of (performance|execution)", text):
        add("Place of cause of action, party residences and places of performance", "Needed to fix territorial jurisdiction under CPC Sections 16 to 20 or Article 226(2).", "critical")
    if not re.search(r"petitioner|respondent|plaintiff|defendant|claimant|appellant|party|parties", text):
        add("Full identity, capacity and addresses of all parties", "Needed for cause title, locus standi and service.", "important")
    if "arbitrat" in text and not re.search(r"\bseat\b|\bvenue\b", text):
        add("Arbitration seat and venue, and the full arbitration clause", "Seat decides supervisory courts under the Arbitration and Conciliation Act, 1996.", "critical")
    if "arbitrat" in text and not re.search(r"notice|section 21|appointment", text):
        add("Notice invoking arbitration and appointment mechanism", "Needed to show the arbitration was validly invoked.", "important")
    if re.search(r"article 226|article 32|writ", text) and not re.search(r"alternative remedy|fundamental right|state|instrumentality|public authority|article 12", text):
        add("State action facts and availability of an alternative remedy", "Maintainability of a writ depends on Article 12 status and the efficacious alternative remedy rule.", "critical")
    if re.search(r"article 136|\bslp\b|special leave", text) and not re.search(r"impugned (order|judgment)|certified copy|pronounced", text):
        add("Impugned order or judgment, date of pronouncement and certified copy", "Needed for limitation and for an SLP under Article 136.", "critical")
    if re.search(r"\bfir\b|offence|\bbns\b|accused|police", text) and not re.search(r"fir no|crime no|case no|police station|\bps\b", text):
        add("FIR number, police station, date and sections invoked under BNS", "Needed to test quashing, bail or cognisability under BNSS.", "important")
    if re.search(r"indian penal code|\bipc\b|code of criminal procedure|\bcrpc\b|indian evidence act", text):
        add("Date of the alleged offence or proceeding", "Decides whether BNS, BNSS and BSA (in force from 1 July 2024) or the repealed statutes apply.", "critical")
    return items


def similarity(a, b):
    """Keyword-overlap scores for two gap descriptions: (jaccard, containment, shared_keyword_count)."""
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return 0.0, 0.0, 0
    shared = len(ka & kb)
    return shared / len(ka | kb), shared / min(len(ka), len(kb)), shared


def _cosine(u, v):
    num = sum(x * y for x, y in zip(u, v))
    den = (sum(x * x for x in u) ** 0.5) * (sum(y * y for y in v) ** 0.5)
    return num / den if den else 0.0


def is_duplicate(a, b, embed_fn=None):
    """True when two gap descriptions ask for the same evidence.

    1. Jaccard >= JACCARD_MATCH, or
    2. containment >= OVERLAP_MATCH with at least OVERLAP_MIN_WORDS keywords in the shorter item, or
    3. (only if embed_fn is given) the lexical score is borderline AND embedding cosine >= EMBED_MATCH.
    embed_fn(list_of_texts) -> list_of_vectors, e.g. embeddings.embed_documents."""
    jaccard, containment, shared = similarity(a, b)
    if jaccard >= JACCARD_MATCH:
        return True
    shorter = min(len(keywords(a)), len(keywords(b)))
    if shorter >= OVERLAP_MIN_WORDS and containment >= OVERLAP_MATCH:
        return True
    if embed_fn is not None and shared >= 1 and max(jaccard, containment * 0.5) >= EMBED_LEXICAL_FLOOR:
        try:
            va, vb = embed_fn([a, b])
            return _cosine(va, vb) >= EMBED_MATCH
        except Exception:
            return False
    return False


def merge_rfae(rule_items, agent_items, embed_fn=None):
    """Merge LLM-proposed gaps into the rule/NER gaps, dropping semantic duplicates.

    When an LLM item duplicates an existing one, the existing item is kept (its wording is more precise) but takes
    the higher severity and, if it had no explanation, the LLM's explanation."""
    merged = [dict(item) for item in rule_items]
    for entry in agent_items:
        if not isinstance(entry, dict):
            continue
        item = str(entry.get("item", "")).strip()
        if not item:
            continue
        severity = str(entry.get("severity", "important")).lower()
        if severity not in SEVERITY_RANK:
            severity = "important"
        why = str(entry.get("why", "")).strip()
        twin = next((m for m in merged if is_duplicate(m["item"], item, embed_fn)), None)
        if twin is not None:
            if SEVERITY_RANK[severity] < SEVERITY_RANK[twin["severity"]]:
                twin["severity"] = severity
            if not twin.get("why") and why:
                twin["why"] = why
            continue
        merged.append({"item": item, "why": why, "severity": severity, "source": "agent"})
    return sorted(merged, key=lambda entry: SEVERITY_RANK[entry["severity"]])


def readiness(gaps):
    """Plain counts plus a coarse status. No invented percentage: the status is a rule, not a probability."""
    counts = {level: sum(1 for g in gaps if g["severity"] == level) for level in SEVERITY_RANK}
    if counts["critical"]:
        status = "Not ready to draft: critical facts are missing"
    elif counts["important"]:
        status = "Draftable with caveats: important facts are missing"
    else:
        status = "No missing facts detected by the rules (a human reviewer should still confirm)"
    return {"counts": counts, "status": status, "total": len(gaps)}
