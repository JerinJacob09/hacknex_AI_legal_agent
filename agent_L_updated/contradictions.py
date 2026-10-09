"""Cross-document contradiction detection.

Stage 1 (deterministic, no LLM): find sentence pairs that talk about the same thing (high keyword overlap) but give
different dates / amounts / numbers, or that differ only by negation. Stage 2 (optional, uses the existing local
model): ask it to confirm each candidate. Every result carries both quotes with [document, location] so a reviewer
can check it in seconds."""
import json
import re
from collections import defaultdict

import entities as ent
from textutil import keywords, normalize_space

NEGATIONS = re.compile(r"\b(?:not|no|never|neither|nor|without|cannot|can't|isn't|wasn't|didn't|hasn't|haven't|shall not|none)\b", re.I)
MIN_SHARED_KEYWORDS = 3
TOPIC_JACCARD = 0.5
MAX_SENTENCES = 6000
MAX_POSTINGS = 300  # words appearing in more sentences than this carry no topic signal and are not indexed


def _facts(sentence):
    dates, rest = ent.extract_dates(sentence)
    rest = ent.CITE_RE.sub(" ", rest)
    rest = ent.PROV_RE.sub(" ", rest)
    rest = re.sub(r"^\s*\(?\d{1,2}[.)]\s+", "", rest)
    values = ent.extract_values(rest)
    topic = frozenset(k for k in keywords(rest) if not k.isdigit())
    return {"dates": dates, "values": values, "topic": topic, "negated": bool(NEGATIONS.search(sentence))}


def _units(chunks):
    units = []
    for chunk in chunks:
        for sentence in ent.split_sentences(chunk.text):
            plain = normalize_space(sentence)
            if len(plain.split()) < 5:
                continue
            facts = _facts(plain)
            if len(facts["topic"]) >= MIN_SHARED_KEYWORDS:
                units.append((chunk, plain, facts))
            if len(units) >= MAX_SENTENCES:
                return units
    return units


def _disjoint_conflict(a, b):
    """Both sides state values of the same kind and none of them agree."""
    return bool(a) and bool(b) and not (a & b)


def find_candidates(chunks, topic_jaccard=TOPIC_JACCARD):
    units = _units(chunks)
    index = defaultdict(list)
    for i, (_, _, facts) in enumerate(units):
        for word in facts["topic"]:
            index[word].append(i)
    seen, candidates = set(), []
    for i, (chunk_a, text_a, fa) in enumerate(units):
        partners = defaultdict(int)
        for word in fa["topic"]:
            postings = index[word]
            if len(postings) > MAX_POSTINGS:
                continue
            for j in postings:
                if j > i:
                    partners[j] += 1
        for j, shared in partners.items():
            if shared < MIN_SHARED_KEYWORDS:
                continue
            chunk_b, text_b, fb = units[j]
            if chunk_a.id == chunk_b.id and text_a == text_b:
                continue
            union = len(fa["topic"] | fb["topic"])
            if union == 0 or shared / union < topic_jaccard:
                continue
            kind = detail = None
            if _disjoint_conflict(fa["dates"], fb["dates"]) and not (fa["dates"] & fb["dates"]):
                kind, detail = "date", f"{', '.join(sorted(fa['dates']))} vs {', '.join(sorted(fb['dates']))}"
            elif not fa["dates"] and not fb["dates"] and _disjoint_conflict(fa["values"], fb["values"]):
                kind, detail = "number", f"{', '.join(sorted(fa['values']))} vs {', '.join(sorted(fb['values']))}"
            elif (fa["negated"] != fb["negated"] and shared / min(len(fa["topic"]), len(fb["topic"])) >= 0.8
                  and fa["dates"] == fb["dates"] and fa["values"] == fb["values"]):
                kind, detail = "negation", "one statement negates the other"
            if kind is None:
                continue
            key = (chunk_a.id, text_a[:60], chunk_b.id, text_b[:60])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "kind": kind, "detail": detail, "confirmed": None, "explanation": "",
                "a": {"doc": chunk_a.doc, "loc": chunk_a.label(), "chunk_id": chunk_a.id, "quote": text_a},
                "b": {"doc": chunk_b.doc, "loc": chunk_b.label(), "chunk_id": chunk_b.id, "quote": text_b},
                "cross_document": chunk_a.doc != chunk_b.doc,
            })
    candidates.sort(key=lambda c: (not c["cross_document"], c["kind"] != "date"))
    return candidates


JUDGE_PROMPT = (
    "Two statements come from a legal case file. Decide whether they CONTRADICT each other about the same fact "
    "(same event, party, obligation or amount), as opposed to describing two different things.\n"
    "Respond with ONLY JSON: {{\"contradiction\": true or false, \"explanation\": \"one sentence\"}}.\n\n"
    "STATEMENT A ({la}):\n{qa}\n\nSTATEMENT B ({lb}):\n{qb}"
)


def verify_candidates(candidates, llm_fn, limit=25):
    """Ask the local model to confirm candidates. llm_fn(prompt) -> str. Candidates beyond `limit` stay unverified."""
    out = []
    for n, cand in enumerate(candidates):
        cand = dict(cand)
        if n < limit and llm_fn is not None:
            try:
                reply = llm_fn(JUDGE_PROMPT.format(la=cand["a"]["loc"], qa=cand["a"]["quote"], lb=cand["b"]["loc"], qb=cand["b"]["quote"]))
                match = re.search(r"\{.*\}", reply or "", re.S)
                verdict = json.loads(match.group(0)) if match else None
            except Exception:
                verdict = None
            if isinstance(verdict, dict) and "contradiction" in verdict:
                cand["confirmed"] = bool(verdict["contradiction"])
                cand["explanation"] = str(verdict.get("explanation", "")).strip()
        out.append(cand)
    return out


def report(candidates, include_rejected=False):
    """Candidates the model rejected are hidden by default; unverified ones stay (marked as such)."""
    return [c for c in candidates if include_rejected or c["confirmed"] is not False]
