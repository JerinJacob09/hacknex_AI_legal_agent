"""Deterministic extraction of the things a legal claim can get wrong: dates, amounts, provisions, citations, names."""
import re

from textutil import normalize_space

_MONTH_NAMES = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

DATE_NUM_RE = re.compile(r"(?<![\w/])(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})(?![\w/])")
DATE_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+day)?(?:\s+of)?\s+({_MONTH_NAMES})\.?,?\s+(\d{{4}})\b", re.I)
DATE_MDY_RE = re.compile(rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
DATE_MY_RE = re.compile(rf"\b({_MONTH_NAMES})\.?,?\s+(\d{{4}})\b", re.I)

CITE_RE = re.compile(
    r"(?:\(\d{4}\)\s*\d+\s*SCC(?:\s*\((?:Cri|L&S|Civ|Tax)\))?\s*\d+"
    r"|\[\d{4}\]\s*\d+\s*SCR\s*\d+|\(\d{4}\)\s*\d+\s*SCR\s*\d+"
    r"|AIR\s*\d{4}\s*[A-Za-z]{2,6}\s*\d+"
    r"|\d{4}\s*SCC\s*OnLine\s*[A-Za-z]+\s*\d+"
    r"|MANU/[A-Z]+/\d+/\d{4})",
    re.IGNORECASE,
)

_NUM = r"\d+[A-Za-z]{0,2}(?:\(\w+\))*"
PROV_RE = re.compile(rf"\b(Sections?|Articles?|Art\.|Sec\.)\s*({_NUM}(?:\s*(?:,|and|&|to|or)\s*{_NUM})*)", re.IGNORECASE)

_MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
         "thousand": 1e3, "million": 1e6, "billion": 1e9, "mn": 1e6, "bn": 1e9}
VALUE_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*(lakhs?|lacs?|crores?|cr|thousand|million|billion|mn|bn)\b)?", re.I)

TAG_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.I)

_PARTY = r"[A-Z][\w.&'-]*(?:\s+(?:[A-Z][\w.&'-]*|of|and|the|for|&))*"
CASE_RE = re.compile(rf"\b({_PARTY})\s+(?:v\.?|vs\.?|versus)\s+({_PARTY})")

_CONNECTORS = {"of", "and", "the", "for", "&"}
_PARTY_NOISE = {"ors", "ors.", "anr", "anr.", "state", "union", "india", "indian", "of", "and", "the", "for", "&", "limited", "ltd", "ltd.",
                "pvt", "pvt.", "private", "co", "co.", "company", "corporation", "inc", "llp"}

_LEAD_NOISE = {"the", "this", "that", "these", "those", "in", "on", "at", "under", "as", "by", "it", "he", "she", "they", "however",
               "therefore", "further", "accordingly", "held", "section", "sections", "article", "articles", "clause", "rule", "order",
               "schedule", "exhibit", "annexure", "according", "also", "thus", "hence", "but", "and", "if", "when", "while", "where",
               "per", "see", "for", "from", "with", "to", "a", "an", "all", "any", "such", "both", "each", "court", "petitioner",
               "respondent", "plaintiff", "defendant", "appellant", "party", "parties", "agreement", "contract", "notice"}
_STATUTE_END = {"act", "code", "sanhita", "adhiniyam", "constitution", "rules", "regulations", "schedule", "order", "amendment"}
_GENERIC_CAPS = {"supreme", "high", "district", "court", "tribunal", "bench", "india", "indian", "government", "union", "state", "section",
                 "article", "clause", "act", "code", "constitution", "sanhita", "adhiniyam", "commission", "council", "board", "authority"}
_TRAIL_NOISE = {"court", "bench", "tribunal", "held", "observed", "also", "further", "judge", "applies", "and", "the", "of", "for", "&"}
_CORP_SUFFIX = {"pvt", "ltd", "private", "limited", "co", "inc", "llp", "company", "corporation", "corp"}
_TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Smt|Shri|Sri|Dr|M/s|Justice|Hon'ble)\.?\s+((?:[A-Z][\w.'-]+\s?){1,3})")
_CAP_RUN_RE = re.compile(r"(?:[A-Z][A-Za-z.&'-]*\s+)+[A-Z][A-Za-z.&'-]*")
_DAYS_MONTHS = set(MONTHS) | {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

_ABBR = {"v", "vs", "no", "nos", "sec", "secs", "art", "arts", "rs", "mr", "mrs", "ms", "dr", "smt", "shri", "hon", "ltd", "pvt", "co",
         "inc", "cl", "para", "paras", "ors", "anr", "ch", "vol", "pp", "p", "viz", "etc", "cf", "ex", "dt", "ie", "eg", "i.e", "e.g", "u/s", "m/s"}


def _iso(day, month, year):
    try:
        day, month, year = int(day), int(month), int(year)
    except ValueError:
        return None
    if year < 100:
        year += 2000 if year <= 50 else 1900
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1000 <= year <= 2999):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_dates(text):
    """Return (set of ISO dates or YYYY-MM month-years, text with date spans blanked out)."""
    found = set()

    def blank(pattern, builder, source):
        def repl(match):
            iso = builder(match)
            if iso is None:
                return match.group(0)
            found.add(iso)
            return " "
        return pattern.sub(repl, source)

    text = blank(DATE_NUM_RE, lambda m: _iso(m.group(1), m.group(2), m.group(3)), text or "")
    text = blank(DATE_DMY_RE, lambda m: _iso(m.group(1), MONTHS[m.group(2).lower()], m.group(3)), text)
    text = blank(DATE_MDY_RE, lambda m: _iso(m.group(2), MONTHS[m.group(1).lower()], m.group(3)), text)
    text = blank(DATE_MY_RE, lambda m: f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}", text)
    return found, text


def date_supported(date, reference_dates):
    if date in reference_dates:
        return True
    if len(date) == 7:  # month-year only: any full date in that month supports it
        return any(ref.startswith(date) for ref in reference_dates)
    return False


def _canon(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def extract_values(text, lenient=False):
    """Numbers and amounts as canonical strings; '5 lakh', 'Rs. 5,00,000' and '500000' all become '500000'.

    lenient=True (used for the *source* side) also keeps the bare multiplier number, so '5 lakh' in a source
    supports a claim that says '5'. Claim side stays strict."""
    out = set()
    for match in VALUE_RE.finditer(text or ""):
        raw = match.group(1).replace(",", "").rstrip(".")
        try:
            value = float(raw)
        except ValueError:
            continue
        mult = _MULT.get((match.group(2) or "").lower(), 1)
        out.add(_canon(value * mult))
        if mult != 1 and lenient:
            out.add(_canon(value))
    return out


def provisions(text):
    found = set()
    for match in PROV_RE.finditer(text or ""):
        kind = "Article" if match.group(1).lower().startswith("art") else "Section"
        for number in re.findall(_NUM, match.group(2)):
            found.add(f"{kind} {number.lower()}")
    return found


def norm_cite(citation):
    return re.sub(r"\s+", "", citation).upper()


def citations(text):
    return {norm_cite(c) for c in CITE_RE.findall(text or "")}


def _party_tokens(party):
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", party)
    return {w.lower().strip(".'-") for w in words if w.lower().strip(".'-") not in _PARTY_NOISE and len(w.strip(".'-")) > 1}


def case_names(text):
    """Return list of (display_name, party_token_set_left, party_token_set_right)."""
    out = []
    for match in CASE_RE.finditer(text or ""):
        left, right = match.group(1).split(), match.group(2).split()
        while left and left[0].lower().strip(".") in _LEAD_NOISE:
            left.pop(0)
        while left and left[-1].lower() in _CONNECTORS:
            left.pop()
        while right and right[-1].lower().strip(".") in _TRAIL_NOISE:
            right.pop()
        left, right = " ".join(left), " ".join(right)
        lt, rt = _party_tokens(left), _party_tokens(right)
        if lt and rt:
            out.append((f"{left} v. {right}", lt, rt))
    return out


def party_names(text):
    """Capitalised multi-word names (people, firms). A heuristic, deliberately conservative."""
    names = set()
    for match in _TITLE_RE.finditer(text or ""):
        names.add(normalize_space(match.group(1)).strip(" .'-"))
    for match in _CAP_RUN_RE.finditer(text or ""):
        words = match.group(0).split()
        while words and words[0].lower().strip(".") in _LEAD_NOISE:
            words.pop(0)
        if len(words) < 2:
            continue
        lowered = [w.lower().strip(".,") for w in words]
        if lowered[-1] in _STATUTE_END or any(w in _STATUTE_END for w in lowered):
            continue
        if all(w in _DAYS_MONTHS or w in _GENERIC_CAPS for w in lowered):
            continue
        while lowered and lowered[0] in _GENERIC_CAPS:
            words.pop(0)
            lowered.pop(0)
        if len(words) >= 2:
            names.add(" ".join(words))
    return {n for n in names if n}


def name_tokens(name):
    return {w.lower().strip(".,'-") for w in re.findall(r"[A-Za-z][A-Za-z.'-]*", name)
            if w.lower().strip(".,'-") not in _CORP_SUFFIX and len(w.strip(".,'-")) > 1}


def strip_tags(text):
    return normalize_space(TAG_RE.sub(" ", text or ""))


def tag_ids(text):
    ids = []
    for match in TAG_RE.finditer(text or ""):
        ids.extend(part.strip().upper() for part in re.split(r"[,;]", match.group(1)))
    return ids


def split_sentences(text):
    """Sentence splitter that respects legal abbreviations ('v.', 'Sec.', 'Rs.') and keeps [S1] tags with their sentence."""
    text = TAG_RE.sub(lambda m: "[" + m.group(1).upper() + "]", text or "")
    text = re.sub(r"([.!?])\s*((?:\[S\d+(?:\s*[,;]\s*S\d+)*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)
    sentences = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        start = 0
        for match in re.finditer(r"[.!?]+(?=\s+(?:[A-Z\[(\"“]|\d))", line):
            before = line[start:match.end()]
            last = before.split()[-1].lower().rstrip(".!?") if before.split() else ""
            if last in _ABBR or re.fullmatch(r"[a-z]", last) or re.fullmatch(r"\(?\d{1,2}\)?", last) or re.fullmatch(r"(?:[a-z]\.){1,}[a-z]?", last + "."):
                continue
            sentences.append(before.strip())
            start = match.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences
