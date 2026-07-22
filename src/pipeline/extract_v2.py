"""extract_v2 - integrated field extractor for the MIB Doc Challenge.

Pipeline
--------
    visible OCR lines (cached, TRUSTED)
      -> page typing (6 packet page types, fuzzy header match)
      -> row grouping + dual-mode label anchoring (inline "Label:value" AND
         geometric two-column label/value)
      -> cross-page resolution by FIELD_MANUAL evidence precedence
      -> vocab.snap()   closed-vocabulary fields
      -> repair.repair() free-text fields (visible-only spacing repair)
      -> flags.detect_flags() for risk_flags

Why this beats the baseline label-anchoring
-------------------------------------------
The baseline only understood the *geometric* layout (a label line with its value
in a column to the right).  Surveying TRAIN shows the packet generator emits two
completely different layouts, and four of the six page types use the other one:

    registry extract / intake form :  ['Species Code']  ['VENUSIAN_MYCELIAL']
    fee receipt / slip / sponsor   :  ['Species Match:JOVIAN_GASFORM']

`norm("Fee Status:paid") == "feestatuspaid"`, which matched no alias, so
fee_status fell through to 'unknown' on every fee receipt that used the inline
layout - the single largest hole in the baseline.  extract_v2 parses both.

Both anchor modes are fuzzy (difflib over an alnum-squashed string), because the
labels themselves arrive shredded: 'Pupose filid repair', 'ponsorID:SPN-6666',
'CoseID', 'Feo Status:unpald', 'Arival Date'.  A literal alias set cannot see
those.

Evidence precedence (FIELD_MANUAL "Trusted Evidence", ranks 1-6)
---------------------------------------------------------------
    1  adjudicator stamp / signed manual correction
    2  intake form fields              (FORM I-8090, MIB Fee Receipt)
    3  biometric slip                  (FORM B-13)
    4  sponsor attestation letter
    5  registry extract
    6  machine-readable text layer     -- NEVER a value source here

Precedence is applied as a *weight in a consensus vote*, not as a strict
tie-break.  Strict "best rank wins outright" was measurably the wrong
aggregator: on TRAIN the highest-precedence page is also the most damaged one,
so the intake form carries a name matching truth on only 79% of packets while
the slip manages 89% and the registry 87%.  Strict precedence faithfully
propagates the intake page's OCR garbage.  Weighted consensus keeps the policy
ordering as a prior while letting agreement between independent pages overrule
a mangled top-ranked read.

The ordering still does the work the policy intends.  On the 48 TRAIN packets
where the intake and sponsor names genuinely conflict -- the "sponsor letter
names a different applicant" trap -- the intake name is truth 30 times and the
sponsor name 18, and the 0.70 sponsor weight reproduces that without any
special case.

Rank 1 ("Manual correction: sponsor is SPN-4705.") is weighted to outvote any
achievable consensus of ordinary pages, because a signed correction exists
precisely to overrule the forms.  The parser fires on 109 TRAIN packets and
agrees with the label 99.1% of the time.

Multi-applicant packets (PRD.md:54 / FIELD_MANUAL.md:80)
--------------------------------------------------------
A page that states a Case ID materially different from the active case_id is
demoted below every in-packet source rather than deleted, so a packet whose
pages are ALL mis-OCR'd still yields values.  "Materially different" means edit
distance >= 2 on the digit payload after OCR-confusion folding, because a
distance-1 difference is overwhelmingly OCR noise (measured on TRAIN:
'MIB-000048' printed on a page of packet MIB-000046 is the 6->8 confusion, not a
second applicant).

Redaction markers
-----------------
Values like '[NAME CUT OUT]', '[REGISTRY LOST]', '[SPECIES WHITEOUT]' are
dropped as candidates rather than emitted, letting a lower-precedence page fill
the field.  If nothing fills it the field stays at its placeholder; the scorer
removes genuinely unrecoverable fields from that case's maximum.

Safety
------
hidden_text is never read by this module, and (since the DEFECT-1 fix) never by
repair.py either: spacing templates come only from trusted visible OCR, so the
untrusted text layer cannot influence a scored value's characters OR its
whitespace.  Nothing is keyed to a
case_id; every alias, threshold and page-type pattern is generic packet
vocabulary, not data-derived from labels.
"""
from __future__ import annotations

import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import flags as flags_mod  # noqa: E402
import repair as repair_mod  # noqa: E402
import vocab  # noqa: E402

CONTENT_FIELDS = ("applicant_name", "species_code", "home_world", "visa_class",
                  "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
                  "fee_status")

PLACEHOLDER = {
    "applicant_name": "unknown", "species_code": "unknown", "home_world": "unknown",
    "visa_class": "unknown", "sponsor_id": "SPN-0000", "arrival_date": "1900-01-01",
    "declared_purpose": "unknown", "risk_flags": "none", "fee_status": "unknown",
}

# ---------------------------------------------------------------- page typing

# rank = FIELD_MANUAL "Trusted Evidence" precedence (lower wins)
PAGE_PATTERNS = [
    ("note",     2, ("manualadjudicatornote", "adjudicatornote")),
    ("intake",   2, ("formi8090", "extraterrestrialworkauthorization", "primaryintakerecord")),
    ("fee",      2, ("mibfeereceipt", "feereceipt")),
    ("slip",     3, ("formb13", "biometricscanslip")),
    ("sponsor",  4, ("sponsorattestationletter", "attestationletter")),
    ("registry", 5, ("planetaryregistryextract", "registryextract")),
]
PAGE_RANK = {"note": 2, "intake": 2, "fee": 2, "slip": 3, "sponsor": 4,
             "registry": 5, "other": 6}
FOREIGN_RANK = 9  # a page belonging to a different applicant

HEADER_THR = 0.68

# ---------------------------------------------------------------- label sets

FIELD_ALIASES = {
    "applicant_name": ("applicant", "applicantname", "registryname", "name",
                       "applicantnamе", "intakename"),
    "species_code":   ("speciescode", "speciesmatch", "species"),
    "home_world":     ("homeworld", "worldoforigin", "planetoforigin", "world"),
    "visa_class":     ("visaclass", "class", "visatype"),
    "sponsor_id":     ("sponsorid", "sponsor"),
    "arrival_date":   ("arrivaldate", "dateofarrival", "arrival"),
    "declared_purpose": ("declaredpurpose", "purpose", "statedpurpose"),
    "fee_status":     ("feestatus", "fee", "paymentstatus"),
}

# Labels that exist on these pages but are NOT fields we extract.  A line is
# skipped when its best match is one of these, which stops e.g. 'Case ID' from
# being dragged into visa_class by 'class', or 'Observed flags' into a value.
DECOY_LABELS = (
    "caseid", "case", "packet", "page", "biometricconfidence", "observedflags",
    "registrystatus", "amount", "waivercode", "finding", "reason", "form",
    "scanimage", "passportimage", "registryimage", "note", "status",
    "syntheticchallengedocument", "mibeyesonly", "issued", "expires",
    "signature", "signed", "stamp", "date", "confidence", "match",
    # page furniture that is a near-prefix of a real alias.  'REGISTRY IMAGE'
    # was being split into label 'REGISTRY' (0.80 against alias 'registryname')
    # and value 'IMAGE', injecting a fake applicant_name on every registry page.
    "registry", "image", "passport", "scan", "primaryintakerecord", "intake",
    "sample", "denial", "archive", "copy", "artifact", "redacted", "casework",
)

LABEL_THR = 0.74          # fuzzy alias match threshold
LABEL_MARGIN = 0.02       # must beat the best decoy by this much

# Values that mean "the evidence was destroyed" - never emit, let a lower
# precedence page answer instead.
REDACTION_RE = re.compile(
    r"(\[|\()?\s*(name|registry|species|world|purpose|sponsor|data|field|value)?\s*"
    r"(cut\s*out|cutout|whiteout|white\s*out|lost|redacted|illegible|missing|"
    r"destroyed|torn|unreadable|removed)\s*(\]|\))?", re.I)


PRIORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "priors_train.json")


def build_priors(force: bool = False) -> dict:
    """Freeze the TRAIN-only majority class for fields we may have to guess.

    Mined from TRAIN_IDS labels exactly like vocab.py mines its vocabularies;
    DEV is never touched and nothing is keyed to a case_id.
    """
    import json
    if os.path.exists(PRIORS_PATH) and not force:
        with open(PRIORS_PATH) as f:
            return json.load(f)
    import collections
    labels = common.load_labels()
    train, _dev = common.split_ids(labels)
    out = {}
    for field in ("fee_status",):
        c = collections.Counter(labels[i][field] for i in train
                                if labels[i].get(field))
        top, n = c.most_common(1)[0]
        out[field] = {"value": top, "share": n / sum(c.values()),
                      "n_train": sum(c.values())}
    with open(PRIORS_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return out


try:
    PRIORS = build_priors()
except Exception:
    PRIORS = {"fee_status": {"value": "paid", "share": 0.0, "n_train": 0}}


# --- READ THIS BEFORE CHANGING -------------------------------------------
# When a packet contains NO visible fee evidence at all (~32% of TRAIN: no fee
# receipt page and no fee text anywhere), fee_status can only be guessed.  The
# TRAIN majority class is 'paid' (66.6%).  Emitting it is a PRIOR, not evidence.
#
# Measured A/B on DEV, holding everything else fixed:
#
#            fee_acc   extraction   classification   false_approvals   TOTAL
#   ON        0.815      43.92          52.10              29          108.14
#   OFF       0.635      43.12          53.75              19          109.05
#
# So the prior BUYS 0.80 extraction points and COSTS 1.65 classification points
# and 10 extra catastrophic false approvals, because the stub classifier reads
# fee_status=='unknown' as a NEEDS_REVIEW trigger and a guessed 'paid' removes
# it.  Net -0.91 total under that classifier.
#
# The paragraph above was a DEV measurement against a stub classifier, and the
# setting was left ON on the ARGUMENT (untested at the time) that a real
# classifier consuming fee_is_prior / fee_observed would route unverified fee
# statuses to NEEDS_REVIEW and keep both gains.
#
# RE-MEASURED ON TRAIN, out of fold, with the shipped classifier
# (pipeline/_fin_feeprior.py: both settings' features rebuilt from scratch,
# 5-fold OOF over fold seeds 0 and 1, each variant given its own selected
# decision layer, TRAIN OOF rows scored end-to-end by the real evaluate.py):
#
#            extraction   classification   calib   false_approvals   TOTAL/150
#   ON          43.35          62.61       15.78       22 / 800       121.74
#   OFF         42.23          63.55       15.50       28 / 800       121.27
#
# The argument holds: with a provenance-aware classifier the prior buys +1.12
# extraction for -0.94 classification (net +0.47) and REDUCES false approvals
# 28 -> 22 instead of adding ten of them.  DEV, reported afterwards and not used
# to decide, agrees (124.22 ON vs 123.25 OFF).
#
# So this stays True - now by measurement rather than by inertia.  If a future
# downstream model does NOT consume fee_is_prior / fee_observed, re-run
# _fin_feeprior.py before trusting this line.
USE_FEE_PRIOR = True


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_redacted(v: str) -> bool:
    s = str(v).strip()
    if not s:
        return True
    if "[" in s or "]" in s:
        return True
    core = _alnum(s)
    if not core or len(core) < 2:
        return True
    if REDACTION_RE.fullmatch(s.strip()):
        return True
    return False


# ---------------------------------------------------------------- page typing

def page_type(page_lines) -> str:
    """Fuzzy-match the page's furniture against the six known packet page types."""
    heads = [_alnum(l["t"]) for l in page_lines[:6]]
    joined = " ".join(_alnum(l["t"]) for l in page_lines)
    best, best_r = "other", 0.0
    for name, _rank, pats in PAGE_PATTERNS:
        for pat in pats:
            for h in heads:
                if not h:
                    continue
                # sliding prefix match: headers get suffixed with junk
                r = _ratio(h[:len(pat) + 4], pat)
                if r > best_r:
                    best, best_r = name, r
            if pat in joined and best_r < 0.99:
                best, best_r = name, 0.99
    return best if best_r >= HEADER_THR else "other"


# ------------------------------------------------------- case-id ownership

_ID_RE = re.compile(r"[MN][I1L][B8][-_ ]?([0-9OQIlSB]{4,8})", re.I)
_DIGIT_FOLD = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0",
                             "I": "1", "l": "1", "i": "1", "|": "1",
                             "S": "5", "s": "5", "B": "8", "Z": "2", "G": "6"})


def _id_digits(s: str) -> str:
    return s.translate(_DIGIT_FOLD).lstrip("0") or "0"


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def page_is_foreign(page_lines, own_digits: str) -> bool:
    """True when this page states a Case ID that is materially another case.

    Only the *stated* Case ID counts (the 'Packet MIB-x / page n' footer is
    generator furniture and always matches).  Distance-1 is treated as OCR
    noise, not a second applicant.
    """
    stated = []
    for l in page_lines:
        t = l["t"]
        low = _alnum(t)
        if low.startswith("packet") or "page" in low[-8:]:
            continue
        for m in _ID_RE.finditer(t):
            stated.append(_id_digits(m.group(1)))
    if not stated:
        return False
    # a page is foreign only if EVERY id it states is far from ours
    return all(_lev(d, own_digits) >= 2 for d in stated)


# ---------------------------------------------------------------- row grouping

def group_rows(page_lines):
    """Cluster a page's lines into visual rows by vertical overlap."""
    ls = sorted(page_lines, key=lambda l: ((l["y0"] + l["y1"]) / 2, l["x0"]))
    rows, cur = [], []
    for l in ls:
        if not cur:
            cur = [l]
            continue
        h = max(min(l["y1"] - l["y0"], cur[-1]["y1"] - cur[-1]["y0"]), 8)
        cy, py = (l["y0"] + l["y1"]) / 2, (cur[-1]["y0"] + cur[-1]["y1"]) / 2
        if abs(cy - py) <= 0.6 * h:
            cur.append(l)
        else:
            rows.append(sorted(cur, key=lambda z: z["x0"]))
            cur = [l]
    if cur:
        rows.append(sorted(cur, key=lambda z: z["x0"]))
    return rows


# ------------------------------------------------------------ label matching

def _best_label(text: str):
    """Return (field, ratio) for the best field-alias match of a label string.

    Returns (None, r) when a decoy label explains the string better.
    """
    key = _alnum(text)
    if not key or len(key) > 40:
        return None, 0.0
    best_f, best_r = None, 0.0
    for f, aliases in FIELD_ALIASES.items():
        for a in aliases:
            r = _ratio(key, a)
            if r > best_r:
                best_f, best_r = f, r
    dec_r = max((_ratio(key, d) for d in DECOY_LABELS), default=0.0)
    if best_r < LABEL_THR or best_r < dec_r + LABEL_MARGIN:
        return None, best_r
    return best_f, best_r


def _split_inline(text: str):
    """Parse an inline 'Label: value' line -> (field, value, ratio) or None.

    Handles a missing/ mangled colon by also trying every whitespace boundary in
    the first few tokens ('Pupose filid repair' -> purpose / 'filid repair').
    """
    cands = []
    for m in re.finditer(r"[:;=]", text):
        cands.append((text[:m.start()], text[m.end():]))
    toks = text.split()
    for k in range(1, min(4, len(toks))):
        cands.append((" ".join(toks[:k]), " ".join(toks[k:])))
    best = None
    for lab, val in cands:
        if not val.strip():
            continue
        f, r = _best_label(lab)
        if f and (best is None or r > best[2]):
            best = (f, val.strip(), r)
    return best


def _row_candidates(row):
    """Yield (field, value, ratio, conf) from one visual row, both layouts."""
    out = []
    # --- inline: each line on its own
    for l in row:
        got = _split_inline(l["t"])
        if got:
            out.append((got[0], got[1], got[2], l["s"]))
    # --- geometric: split the row at its widest horizontal gap
    if len(row) >= 2:
        gaps = [(row[i + 1]["x0"] - row[i]["x1"], i) for i in range(len(row) - 1)]
        gaps.sort(reverse=True)
        for _g, i in gaps[:2]:
            lab = " ".join(x["t"] for x in row[:i + 1])
            val = " ".join(x["t"] for x in row[i + 1:])
            f, r = _best_label(lab)
            if f and val.strip():
                conf = min(x["s"] for x in row[i + 1:])
                out.append((f, val.strip(), r, conf))
            # also allow the label itself to be an inline pair on the left part
            got = _split_inline(lab)
            if got:
                out.append((got[0], got[1], got[2], row[0]["s"]))
    return out


def _looks_like_label(v: str) -> bool:
    """True if this *value* is really another field's label.

    The stacked-layout fallback below is the main source of label bleed: when a
    value column is washed out, the line under 'Applicant' is the next LABEL
    ('Species Code'), not a name.  Measured on TRAIN this was the single largest
    applicant_name error class.
    """
    key = _alnum(v)
    if not key:
        return True
    if _best_label(v)[0] is not None:
        return True
    for f, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if _ratio(key, a) >= 0.82:
                return True
    return any(_ratio(key, d) >= 0.82 for d in DECOY_LABELS)


def _below_candidates(page_lines):
    """Label-only line with its value on the next line down (stacked layout)."""
    out = []
    rows = group_rows(page_lines)
    for i, row in enumerate(rows[:-1]):
        if len(row) != 1:
            continue
        f, r = _best_label(row[0]["t"])
        if not f:
            continue
        nxt = rows[i + 1]
        near = [l for l in nxt if abs(l["x0"] - row[0]["x0"]) < 120]
        if len(near) == 1 and not _looks_like_label(near[0]["t"]):
            out.append((f, near[0]["t"].strip(), r * 0.95, near[0]["s"]))
    return out


# ---------------------------------------------------------- value normalizers

VISA_CODES = ("XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7")
_DATE_RE = re.compile(r"([0-9OQIlSB]{4})\s*[-/. ]\s*([0-9OQIlSB]{1,2})\s*[-/. ]\s*([0-9OQIlSB]{1,2})")


def _clean(v: str) -> str:
    v = str(v).strip()
    v = re.sub(r"^[\s:;=.,\-|]+", "", v)
    v = re.sub(r"[\s:;=,|]+$", "", v)
    return v.strip()


def norm_date(v: str):
    m = _DATE_RE.search(str(v))
    if not m:
        return None
    y, mo, d = (g.translate(_DIGIT_FOLD) for g in m.groups())
    if not (y.isdigit() and mo.isdigit() and d.isdigit()):
        return None
    yi, mi, di = int(y), int(mo), int(d)
    if not (1900 <= yi <= 2200 and 1 <= mi <= 12 and 1 <= di <= 31):
        return None
    return f"{yi:04d}-{mi:02d}-{di:02d}"


def norm_fee(v: str):
    """Order matters: 'unpaid' contains 'paid'."""
    s = _alnum(v)
    if not s:
        return None
    for pat, out in (("unpaid", "unpaid"), ("unpald", "unpaid"), ("unpd", "unpaid"),
                     ("waiv", "waived"), ("waved", "waived"), ("waiver", "waived")):
        if pat in s:
            return out
    if "paid" in s or "pald" in s or "pd" == s:
        return "paid"
    if "unknown" in s or "nomatch" in s:
        return "unknown"
    # fuzzy last resort on the whole token
    best, br = None, 0.0
    for cand in ("unpaid", "waived", "paid"):
        r = _ratio(s[:len(cand) + 2], cand)
        if r > br:
            best, br = cand, r
    return best if br >= 0.80 else None


_AMOUNT_RE = re.compile(r"\$\s*([\d,]+)\s*[.,]\s*(\d\d)")
_WAIVER_RE = re.compile(r"waiver\s*code\W*(.{0,12})", re.I)


def fee_from_receipt(rec):
    """Recover fee_status from a receipt whose Fee Status *value* is destroyed.

    The MIB Fee Receipt prints Amount and WaiverCode next to Fee Status, and on
    TRAIN those two fields determine the answer with no exceptions among the 35
    packets where the Fee Status value itself was unreadable:

        Amount > 0,  WaiverCode N/A   -> paid     (14/14)
        Amount = 0,  WaiverCode N/A   -> unknown  (16/16)
        Amount = 0,  WaiverCode set   -> waived   ( 5/5 )

    This is receipt arithmetic, not a learned correlation: a nonzero charge that
    was not waived is a payment, and a zero charge carrying a waiver code is a
    waiver.  Returns None when the receipt does not show both fields.
    """
    for p in common.pages_of(rec):
        lines = common.page_lines(p)
        if not lines or page_type(lines) != "fee":
            continue
        txt = " ".join(l["t"] for l in lines)
        am = _AMOUNT_RE.search(txt)
        if not am:
            continue
        amount = int(am.group(1).replace(",", "")) + int(am.group(2))
        wm = _WAIVER_RE.search(txt)
        if wm is None:
            continue
        wc = _alnum(wm.group(1))
        waived = bool(wc) and wc not in ("na", "none", "nil", "n", "0")
        if amount > 0 and not waived:
            return "paid"
        if amount == 0 and waived:
            return "waived"
        if amount == 0 and not waived:
            return "unknown"
    return None


def norm_sponsor(v: str):
    """Sponsor ids must have the SPN-#### shape or they are not sponsor ids.

    Without this gate the geometric splitter happily returned 'Attestation
    Letter' (from the 'Sponsor Attestation Letter' page header) as a sponsor id.
    """
    s = str(v)
    m = re.search(r"\b[S5$][PF][NM][-\s.]?([0-9OQIlSB]{4})\b", s, re.I)
    if m:
        d = m.group(1).translate(_DIGIT_FOLD)
        if d.isdigit():
            return f"SPN-{d}"
    rep = vocab.repair_sponsor_id(s)
    if rep and re.fullmatch(r"SPN-\d{4}", rep):
        return rep
    return None


# --------------------------------------------------- rank-1 manual corrections

CORRECTION_STEM = "manualcorrection"


def _alnum_map(s: str):
    """Alnum-squashed string plus, for each squashed char, its index in `s`."""
    key, idx = [], []
    for i, ch in enumerate(s):
        c = ch.lower()
        if c.isalnum():
            key.append(c)
            idx.append(i)
    idx.append(len(s))
    return "".join(key), idx


def correction_candidates(page_lines):
    """Parse 'Manual correction: <field> is <value>.' annotations.

    These are FIELD_MANUAL rank-1 evidence -- a signed manual note that
    explicitly overrides what the forms say -- and they appear on ~14% of TRAIN
    packets carrying sponsor_id, visa_class, fee_status or applicant_name.
    Matching is fuzzy on an alnum-squashed string because the annotation is
    handwritten-styled and OCRs with the spaces dropped
    ('Manualcorrection:sponsorisSPN-4705.').
    """
    out = []
    for l in page_lines:
        raw = l["t"]
        key, idx = _alnum_map(raw)
        if len(key) < len(CORRECTION_STEM) + 4:
            continue
        # locate the stem near the start of the line
        pos, best = None, 0.0
        for p in range(0, min(12, len(key))):
            r = _ratio(key[p:p + len(CORRECTION_STEM)], CORRECTION_STEM)
            if r > best:
                pos, best = p + len(CORRECTION_STEM), r
        if best < 0.80 or pos is None:
            continue
        rest = key[pos:]
        hit = None
        for f, aliases in FIELD_ALIASES.items():
            for a in aliases:
                for L in range(max(3, len(a) - 2), min(len(a) + 3, len(rest) - 2) + 1):
                    if rest[L:L + 2] != "is":
                        continue
                    r = _ratio(rest[:L], a)
                    if r >= 0.80 and (hit is None or r > hit[2]):
                        hit = (f, pos + L + 2, r)
        if hit is None:
            continue
        f, vstart, r = hit
        value = raw[idx[vstart]:] if vstart < len(idx) else ""
        value = re.sub(r"[.\s]+$", "", value.strip())
        if value:
            out.append((f, value, 1.0, l["s"]))
    return out


def norm_visa(v: str):
    up = str(v).upper()
    m = re.search(r"\b(XW[-\s]?[12]|DIP[-\s]?1|MED[-\s]?3|TRANSIT[-\s]?7)\b", up)
    if m:
        return re.sub(r"[\s-]+", "-", m.group(1))
    return None


# ---------------------------------------------------------------- main entry

def _collect_candidates(rec):
    own = _id_digits(str(rec.get("case_id", "")).split("-")[-1])
    cands = {f: [] for f in FIELD_ALIASES}
    page_types = []
    for p in common.pages_of(rec):
        lines = common.page_lines(p)
        if not lines:
            page_types.append("other")
            continue
        ptype = page_type(lines)
        page_types.append(ptype)
        rank = PAGE_RANK[ptype]
        if page_is_foreign(lines, own):
            rank = FOREIGN_RANK
        for row in group_rows(lines):
            for f, val, r, conf in _row_candidates(row):
                cands[f].append((rank, r, conf, _clean(val), ptype, p.get("page", 0)))
        for f, val, r, conf in _below_candidates(lines):
            cands[f].append((rank, r, conf, _clean(val), ptype, p.get("page", 0)))
        # rank-1 signed manual corrections override everything on the packet
        for f, val, r, conf in correction_candidates(lines):
            cands[f].append((1, r, conf, _clean(val), "correction", p.get("page", 0)))
    return cands, page_types


# Evidence-precedence weights used by the consensus resolver.
#
# Strict tier-ordering ("best rank wins outright") turned out to be the wrong
# aggregator.  Measured on TRAIN, the highest-precedence page is also the one
# most often damaged: the intake form carries a name matching truth on only
# 79% of packets, while the biometric slip manages 89% and the registry extract
# 87% -- not because they outrank it, but because when the intake page is torn
# or low-contrast its OCR is garbage and strict precedence propagates that
# garbage.  Weighted consensus keeps precedence as a *prior* and lets agreement
# between independent pages overrule a mangled top-ranked read.
#
# Weights swept on TRAIN.  The sweep is nearly flat except for two effects it
# identifies strongly, both of which match the policy:
#   * the sponsor attestation is discounted (0.70) -- on the 48 TRAIN packets
#     where the intake and sponsor names genuinely conflict, the intake name is
#     the truth 30 times and the sponsor name 18, exactly the FIELD_MANUAL
#     ordering (rank 2 over rank 4);
#   * a page belonging to another applicant is near-vetoed (0.05).
# Rank 1 is weighted to outvote any achievable consensus of ordinary pages: a
# signed manual correction is the top of the FIELD_MANUAL precedence list and
# exists precisely to overrule the forms.
RANK_WEIGHT = {1: 6.00, 2: 1.00, 3: 1.00, 4: 0.70, 5: 0.90, 6: 0.30,
               FOREIGN_RANK: 0.05}
CLUSTER_THR = 0.80
# OCR line confidence enters a cluster's score as conf**CONF_POWER.  Linear
# weighting let two mangled low-confidence reads (0.57 and 0.55) outvote one
# clean read (0.99) purely by count; cubing makes a sub-0.7 line contribute
# almost nothing, which is the right prior for this OCR engine.  Swept on TRAIN:
# weighted accuracy 0.8649 (p=1) -> 0.8669 (p=3) -> 0.8678 (p=6).  The curve is
# monotone but flattening, so 3.0 takes most of the gain without riding the tail.
CONF_POWER = float(os.environ.get("MIB_CONF_POWER", "3.0"))

NORMALIZERS = {"arrival_date": norm_date, "fee_status": norm_fee,
               "visa_class": norm_visa, "sponsor_id": norm_sponsor}


def _resolve(field, entries):
    """Weighted-consensus resolution across pages (see RANK_WEIGHT)."""
    norm = NORMALIZERS.get(field)
    good = []
    for rank, r, conf, val, ptype, pg in entries:
        if _is_redacted(val):
            continue
        v = val
        if norm:
            v = norm(val)
            if v is None:
                continue
        else:
            if len(_alnum(v)) < 2 or _looks_like_label(v):
                continue
        good.append((rank, r, conf, v, ptype, pg))
    if not good:
        return None, None, []

    # Cluster near-identical readings; a cluster's score is the sum of its
    # members' precedence x label-match x OCR confidence.
    clusters = []
    for rank, r, conf, v, ptype, pg in sorted(
            good, key=lambda g: -(RANK_WEIGHT.get(g[0], 0.05) * g[1] * (g[2] ** CONF_POWER))):
        s = RANK_WEIGHT.get(rank, 0.05) * r * (conf ** CONF_POWER)
        key = _alnum(v)
        for c in clusters:
            if norm:
                hit = (v == c["rep"])
            else:
                hit = _ratio(key, c["key"]) >= CLUSTER_THR
            if hit:
                c["score"] += s
                c["n"] += 1
                c["rank"] = min(c["rank"], rank)
                break
        else:
            clusters.append({"key": key, "rep": v, "score": s, "n": 1, "rank": rank})
    best = max(clusters, key=lambda c: (c["score"], c["n"]))
    return best["rep"], best["rank"], good


def extract_fields_v2(rec) -> dict:
    """Extract the 9 content fields from one cached OCR record."""
    if not isinstance(rec, dict):
        # A malformed record must lose its own case at worst, never the run.
        return dict(PLACEHOLDER)
    fields = dict(PLACEHOLDER)
    meta = {}
    cands, page_types = _collect_candidates(rec)

    for f in FIELD_ALIASES:
        value, rank, good = _resolve(f, cands[f])
        meta[f] = {"rank": rank, "n": len(good)}
        if value is not None:
            fields[f] = value

    # --- document-wide regex fallbacks (only when anchoring found nothing)
    text_all = common.visible_text(rec)
    if fields["sponsor_id"] == PLACEHOLDER["sponsor_id"]:
        m = re.search(r"\b[S5][PF]N[-\s]?\d{4}\b", text_all, re.I)
        if m:
            fields["sponsor_id"] = m.group(0).upper()
    if fields["arrival_date"] == PLACEHOLDER["arrival_date"]:
        d = norm_date(text_all)
        if d:
            fields["arrival_date"] = d
    if fields["visa_class"] == PLACEHOLDER["visa_class"]:
        v = norm_visa(text_all)
        if v:
            fields["visa_class"] = v
    fee_source = "anchored"
    if fields["fee_status"] == PLACEHOLDER["fee_status"]:
        fee_source = None
        for l in common.doc_lines(rec):
            if "fee" in _alnum(l["t"]):
                fv = norm_fee(l["t"])
                if fv:
                    fields["fee_status"] = fv
                    fee_source = "line_scan"
                    break
    if fields["fee_status"] == PLACEHOLDER["fee_status"]:
        fv = fee_from_receipt(rec)
        if fv:
            fields["fee_status"] = fv
            fee_source = "receipt_arithmetic"
    if USE_FEE_PRIOR and fields["fee_status"] == PLACEHOLDER["fee_status"]:
        fields["fee_status"] = PRIORS["fee_status"]["value"]
        fee_source = "prior"

    # --- sponsor id shape repair (pattern-based, no vocabulary)
    sp = vocab.repair_sponsor_id(fields["sponsor_id"])
    if sp:
        fields["sponsor_id"] = sp

    # --- closed-vocabulary snapping (TRAIN-mined vocab, guarded)
    fields = vocab.snap_fields(fields, visible_text=text_all)

    # --- free-text word-spacing repair (visible-OCR templates only)
    fields["applicant_name"] = vocab.repair_name(fields["applicant_name"])
    fields = repair_mod.repair_fields(fields, rec)

    # --- risk flags from the dedicated detector
    fields["risk_flags"] = flags_mod.risk_flags_string(rec)

    fields["_meta"] = {"page_types": page_types, "field_meta": meta,
                       "fee_source": fee_source}
    out = {f: fields[f] for f in CONTENT_FIELDS}
    out["_meta"] = fields["_meta"]
    return out


# ---------------------------------------------------------------- features

def extract_features(rec, fields) -> dict:
    """Flat numeric/categorical features for the downstream classifier.

    Deliberately generic: no value is one-hot encoded against a memorised list
    of TRAIN values except the five closed vocabularies, which are policy enums
    from FIELD_MANUAL (visa classes, fee statuses, risk flags) rather than
    data-mined identifiers.  Species/world/name/sponsor contribute only shape
    and resolution statistics, so unseen private values behave identically.
    """
    lines = common.doc_lines(rec)
    confs = [l["s"] for l in lines] or [0.0]
    meta = (fields or {}).get("_meta", {})
    ptypes = meta.get("page_types", [])
    fmeta = meta.get("field_meta", {})

    fl = set(str(fields.get("risk_flags", "none")).split("|")) - {"none", ""}

    feat = {
        # --- document shape
        "n_pages": len(common.pages_of(rec)),
        "n_lines": len(lines),
        "lines_per_page": len(lines) / max(len(common.pages_of(rec)), 1),
        "conf_mean": sum(confs) / len(confs),
        "conf_min": min(confs),
        "conf_p10": sorted(confs)[max(0, len(confs) // 10 - 1)],
        "conf_lt_90_frac": sum(c < 0.90 for c in confs) / len(confs),
        "conf_lt_80_frac": sum(c < 0.80 for c in confs) / len(confs),
        # --- which page types are present (packet completeness)
        **{f"has_page_{t}": int(t in ptypes) for t in
           ("note", "intake", "fee", "slip", "sponsor", "registry", "other")},
        "n_page_types": len(set(ptypes)),
        # --- closed-vocab one-hots (policy enums, not mined identifiers)
        **{f"visa_{c}": int(fields.get("visa_class") == c) for c in VISA_CODES},
        "visa_unknown": int(fields.get("visa_class") in (None, "", "unknown")),
        **{f"fee_{c}": int(fields.get("fee_status") == c)
           for c in ("paid", "unpaid", "waived", "unknown")},
        # --- per-flag booleans
        **{f"flag_{f}": int(f in fl) for f in
           sorted(common.DISQUALIFYING | common.REVIEW_ONLY)},
        "n_flags": len(fl),
        "n_disqualifying": len(fl & common.DISQUALIFYING),
        "n_review_only": len(fl & common.REVIEW_ONLY),
        "flags_confidence": flags_mod.flags_confidence(rec),
        # --- resolution quality per field
        **{f"resolved_{f}": int(fields.get(f) not in (None, "", PLACEHOLDER.get(f)))
           for f in CONTENT_FIELDS},
        **{f"rank_{f}": (fmeta.get(f, {}).get("rank") or 9) for f in FIELD_ALIASES},
        **{f"nev_{f}": fmeta.get(f, {}).get("n", 0) for f in FIELD_ALIASES},
        "n_unknown_fields": sum(fields.get(f) in (None, "", PLACEHOLDER.get(f))
                                for f in CONTENT_FIELDS),
        # fee_status provenance - 'prior' means GUESSED, not observed
        **{f"fee_src_{s}": int(meta.get("fee_source") == s)
           for s in ("anchored", "line_scan", "receipt_arithmetic", "prior")},
        "fee_is_prior": int(meta.get("fee_source") == "prior"),
        "fee_observed": int(meta.get("fee_source") not in (None, "prior")),
        # --- shape checks
        "sponsor_wellformed": int(bool(re.fullmatch(r"SPN-\d{4}",
                                                    str(fields.get("sponsor_id", ""))))),
        "sponsor_placeholder": int(fields.get("sponsor_id") == "SPN-0000"),
        "date_wellformed": int(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}",
                                                 str(fields.get("arrival_date", ""))))),
        "date_placeholder": int(fields.get("arrival_date") == "1900-01-01"),
        "name_n_tokens": len(str(fields.get("applicant_name", "")).split()),
        "species_is_upper_snake": int(bool(re.fullmatch(
            r"[A-Z][A-Z_]{3,}", str(fields.get("species_code", ""))))),
    }

    # --- stamp / annotation tokens visible on the page (rank-1 evidence markers)
    joined = " ".join(_alnum(l["t"]) for l in lines)
    for tok in ("denied", "approved", "needsreview", "rescind", "sampledenial",
                "redacted", "copyartifact", "casework", "archive", "embargo",
                "hardship", "waiver", "diplomaticnote", "revoked"):
        feat[f"tok_{tok}"] = int(tok in joined)

    # --- hidden-vs-visible disagreement (measurement only, never a value source)
    try:
        feat.update({k: v for k, v in
                     repair_mod.disagreement_features(rec, fields).items()
                     if isinstance(v, (int, float))})
    except Exception:
        pass

    # --- cross-document identity consistency
    names = []
    for l in lines:
        g = _split_inline(l["t"])
        if g and g[0] == "applicant_name" and not _is_redacted(g[1]):
            names.append(_alnum(g[1]))
    feat["n_distinct_names"] = len(set(names))
    feat["name_agreement"] = (
        min(_ratio(a, names[0]) for a in names) if names else 0.0)
    return feat


# ---------------------------------------------------------------- self-test

def _fmt_pct(x):
    return f"{100 * x:5.1f}%"


def _main():
    import baseline

    cache = common.load_cache()
    labels = common.load_labels()
    train, dev = common.split_ids(labels)
    dev = [c for c in dev if c in cache]
    train = [c for c in train if c in cache]
    print(f"cache={len(cache)}  train={len(train)}  dev={len(dev)}")
    if len(cache) < 1000:
        print("!! cache incomplete - numbers below are on a biased subset")

    def per_field(rows):
        hits = {f: 0 for f in CONTENT_FIELDS}
        for r in rows:
            t = labels[r["case_id"]]
            for f in CONTENT_FIELDS:
                pv = str(r.get(f, "")).strip().casefold()
                tv = str(t.get(f, "")).strip().casefold()
                pv = " ".join(pv.split())
                tv = " ".join(tv.split())
                if pv == tv:
                    hits[f] += 1
        return {f: hits[f] / max(len(rows), 1) for f in CONTENT_FIELDS}

    def run(fn, tag, ids):
        rows = []
        for cid in ids:
            fields = fn(cache[cid])
            fields = {k: v for k, v in fields.items() if not k.startswith("_")}
            rows.append(baseline.row_for(cid, fields))
        res = common.score_rows(rows, tag)
        return rows, res

    base_rows, base_res = run(baseline.extract_rec, "b_dev", dev)
    v2_rows, v2_res = run(extract_fields_v2, "v2_dev", dev)

    bf, vf = per_field(base_rows), per_field(v2_rows)
    print("\nDEV per-field exact match (n=%d)" % len(dev))
    print(f"{'field':18s} {'w':>2s}  {'baseline':>9s} {'v2':>9s}   delta")
    for f in CONTENT_FIELDS:
        d = vf[f] - bf[f]
        print(f"{f:18s} {common.FIELD_WEIGHTS[f]:2d}  {_fmt_pct(bf[f]):>9s} "
              f"{_fmt_pct(vf[f]):>9s}  {d*100:+6.1f}")
    wsum = sum(common.FIELD_WEIGHTS.values())
    print(f"{'WEIGHTED':18s} {wsum:2d}  "
          f"{_fmt_pct(sum(bf[f]*common.FIELD_WEIGHTS[f] for f in CONTENT_FIELDS)/wsum):>9s} "
          f"{_fmt_pct(sum(vf[f]*common.FIELD_WEIGHTS[f] for f in CONTENT_FIELDS)/wsum):>9s}")

    print("\nDEV baseline : " + common.summarize(base_res))
    print("DEV v2       : " + common.summarize(v2_res))
    print(f"\nextraction   : baseline {base_res['scores']['extraction_score']:.2f}/50"
          f"  ->  v2 {v2_res['scores']['extraction_score']:.2f}/50")

    # sanity: feature vector is flat + numeric
    feats = extract_features(cache[dev[0]], extract_fields_v2(cache[dev[0]]))
    bad = {k: v for k, v in feats.items() if not isinstance(v, (int, float))}
    print(f"\nfeatures: {len(feats)} keys, non-numeric={bad}")
    return v2_res


if __name__ == "__main__":
    _main()
