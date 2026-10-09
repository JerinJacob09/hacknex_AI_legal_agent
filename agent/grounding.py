"""Claim-level grounding: every answer sentence must cite retrieved chunks, and the names, dates, amounts,
provisions and case citations it contains must literally appear in those chunks.

Pure Python, no LLM involved in the checks. The checks are deliberately conservative: a sentence that cannot be
traced to a retrieved chunk is withheld (mode="drop") or marked (mode="flag")."""
import hashlib
import re
from dataclasses import dataclass, field

import entities as ent
from textutil import normalize_space, tokens

MIN_OVERLAP = 0.45
NOT_FOUND = "Not found in the provided documents."


@dataclass
class Chunk:
    id: str
    text: str
    doc: str = ""
    page: int = None
    loc: str = ""
    case_id: str = ""
    kind: str = ""
    citation: str = ""
    score: float = 0.0

    def label(self):
        where = f"p.{self.page}" if self.page else (self.loc or "")
        return f"{self.doc}, {where}".rstrip(", ") if self.doc else (where or self.id)


def make_chunk_id(case_id, doc, page, index, text):
    raw = f"{case_id}|{doc}|{page}|{index}|{text[:80]}"
    return "c-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def number_sources(chunks):
    """Return an ordered dict {'S1': Chunk, ...} used both in the prompt and for checking the answer."""
    return {f"S{i}": chunk for i, chunk in enumerate(chunks, 1)}


def format_sources(sources):
    blocks = []
    for tag, chunk in sources.items():
        blocks.append(f"[{tag}] (document: {chunk.doc or 'unknown'}; location: {chunk.label().split(', ', 1)[-1] if chunk.doc else chunk.loc}; chunk: {chunk.id})\n{chunk.text}")
    return "\n\n".join(blocks)


@dataclass
class Profile:
    dates: set
    values: set
    provisions: set
    numbers: set
    cites: set
    words: set
    stems: set


def profile(text):
    dates, _ = ent.extract_dates(text)
    return Profile(
        dates=dates,
        values=ent.extract_values(text, lenient=True),
        provisions=ent.provisions(text),
        numbers=set(re.findall(r"\d+", text)),
        cites=ent.citations(text),
        words={w.lower().strip(".'-") for w in re.findall(r"[A-Za-z][A-Za-z.'-]*", text)},
        stems=set(tokens(text)),
    )


_PROFILE_CACHE = {}


def cached_profile(text):
    key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    if key not in _PROFILE_CACHE:
        if len(_PROFILE_CACHE) > 2000:
            _PROFILE_CACHE.clear()
        _PROFILE_CACHE[key] = profile(text)
    return _PROFILE_CACHE[key]


def entity_gaps(sentence, reference_text):
    """Entities in `sentence` that do not occur in `reference_text`. Empty list means every checkable fact is present."""
    ref = cached_profile(reference_text)
    gaps = []
    dates, rest = ent.extract_dates(sentence)
    for d in sorted(dates):
        if not ent.date_supported(d, ref.dates):
            gaps.append(f"date {d}")
    for cite in ent.CITE_RE.findall(rest):
        if ent.norm_cite(cite) not in ref.cites:
            gaps.append(f"citation {normalize_space(cite)}")
    rest = ent.CITE_RE.sub(" ", rest)
    for name, left, right in ent.case_names(rest):
        if not (left <= ref.words and right <= ref.words):
            gaps.append(f"case name {name}")
    rest = ent.CASE_RE.sub(" ", rest)
    sentence_provisions = ent.provisions(rest)
    for prov in sorted(sentence_provisions):
        number = re.match(r"\w+\s+(\d+)", prov)
        if prov not in ref.provisions and not (number and number.group(1) in ref.numbers):
            gaps.append(f"provision {prov}")
    rest_no_prov = ent.PROV_RE.sub(" ", rest)
    rest_no_prov = re.sub(r"^\s*\(?\d{1,2}[.)]\s+", "", rest_no_prov)
    for value in sorted(ent.extract_values(rest_no_prov)):
        if value not in ref.values:
            gaps.append(f"number {value}")
    for name in sorted(ent.party_names(rest_no_prov)):
        toks = ent.name_tokens(name)
        if toks and not toks <= ref.words:
            gaps.append(f"name {name}")
    return gaps


def content_overlap(sentence, reference_text):
    words = [t for t in tokens(ent.strip_tags(sentence)) if not t.isdigit()]
    if len(words) < 3:
        return 1.0
    ref = cached_profile(reference_text).stems
    return sum(1 for w in words if w in ref) / len(words)


def is_meta(sentence):
    plain = ent.strip_tags(sentence)
    if len(plain.split()) < 4 and not ent.TAG_RE.search(sentence):
        return True
    if plain.endswith(":") or re.match(r"^#{1,6}\s", plain):
        return True
    if plain.lower().rstrip(". ") in {"not found in documents", NOT_FOUND.lower().rstrip(".")}:
        return True
    return False


@dataclass
class Claim:
    text: str
    tags: list = field(default_factory=list)
    status: str = "unsupported"  # supported | unsupported | meta
    reasons: list = field(default_factory=list)
    sources: list = field(default_factory=list)


def check_claim(sentence, sources, min_overlap=MIN_OVERLAP):
    claim = Claim(text=sentence, tags=ent.tag_ids(sentence))
    if is_meta(sentence):
        claim.status = "meta"
        return claim
    if not claim.tags:
        claim.reasons.append("no source tag")
        return claim
    unknown = [t for t in claim.tags if t not in sources]
    if unknown:
        claim.reasons.append("cites non-existent source " + ", ".join(unknown))
        return claim
    cited = [sources[t] for t in claim.tags]
    claim.sources = [f"{t}: {sources[t].label()} [{sources[t].id}]" for t in claim.tags]
    reference = "\n".join(c.text for c in cited)
    plain = ent.strip_tags(sentence)
    gaps = entity_gaps(plain, reference)
    if gaps:
        claim.reasons.append("not in cited source: " + "; ".join(gaps))
    overlap = content_overlap(plain, reference)
    if overlap < (MIN_OVERLAP if min_overlap is None else min_overlap):
        claim.reasons.append(f"low wording overlap with cited source ({overlap:.2f})")
    if not claim.reasons:
        claim.status = "supported"
    return claim


@dataclass
class GroundedAnswer:
    text: str
    claims: list
    supported: int
    unsupported: int
    withheld: list
    used_sources: list

    @property
    def total(self):
        return self.supported + self.unsupported

    @property
    def supported_share(self):
        return self.supported / self.total if self.total else 1.0


def ground_answer(answer, sources, mode="drop", min_overlap=MIN_OVERLAP):
    """Check each sentence of `answer`. mode='drop' removes unsupported sentences, mode='flag' keeps them with a marker."""
    claims, kept_lines, withheld, used = [], [], [], []
    for line in re.split(r"\n+", answer or ""):
        if not line.strip():
            continue
        pieces = []
        for sentence in ent.split_sentences(line):
            claim = check_claim(sentence, sources, min_overlap)
            claims.append(claim)
            if claim.status == "unsupported":
                withheld.append((claim.text, claim.reasons))
                if mode == "flag":
                    pieces.append(claim.text + " ⚠ [unsupported: " + "; ".join(claim.reasons) + "]")
                continue
            pieces.append(claim.text)
            for tag in claim.tags:
                if tag in sources and tag not in used:
                    used.append(tag)
        if pieces:
            kept_lines.append(" ".join(pieces))
    supported = sum(1 for c in claims if c.status == "supported")
    unsupported = sum(1 for c in claims if c.status == "unsupported")
    text = "\n\n".join(kept_lines).strip()
    if supported == 0 and mode == "drop":
        text = NOT_FOUND + (" (every statement in the model's draft answer failed the source check; see the withheld list)" if unsupported else "")
    return GroundedAnswer(text=text, claims=claims, supported=supported, unsupported=unsupported, withheld=withheld, used_sources=used)


def claim_supported_by_any(sentence, chunks, min_overlap=MIN_OVERLAP):
    """Tag-free check used by the evaluation harness so a baseline answer (which has no tags) can be scored fairly:
    a sentence counts as supported if at least one single retrieved chunk contains all its facts."""
    plain = ent.strip_tags(sentence)
    for chunk in chunks:
        if not entity_gaps(plain, chunk.text) and content_overlap(plain, chunk.text) >= min_overlap:
            return True
    return False


def score_text(text, chunks, min_overlap=MIN_OVERLAP):
    """Return dict(total, supported, unsupported_sentences) for untagged text against retrieved chunks."""
    total = supported = 0
    bad = []
    for sentence in ent.split_sentences(text or ""):
        if is_meta(sentence):
            continue
        total += 1
        if claim_supported_by_any(sentence, chunks, min_overlap):
            supported += 1
        else:
            bad.append(sentence)
    return {"total": total, "supported": supported, "unsupported_sentences": bad}


def fabricated_authorities(text, reference_text):
    """Citations and case names in `text` that do not occur in the retrieved text. Used as the zero-fabrication gate."""
    ref = cached_profile(reference_text)
    bad = []
    for cite in ent.CITE_RE.findall(text or ""):
        if ent.norm_cite(cite) not in ref.cites:
            bad.append(normalize_space(cite))
    stripped = ent.CITE_RE.sub(" ", text or "")
    for name, left, right in ent.case_names(stripped):
        if not (left <= ref.words and right <= ref.words):
            bad.append(name)
    return sorted(set(bad))
