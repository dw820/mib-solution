"""Closed-vocabulary snapping for OCR error correction (MIB Doc Challenge).

Public API
----------
    snap(field, raw_text)                 -> corrected value (or raw_text unchanged)
    snap_fields(fields, visible_text=None)-> dict with every field snapped
    repair_name(raw_text)                 -> open-vocab name repair (camel-split + I/l fix)
    build_vocab(force=False)              -> (re)mine the vocabulary from TRAIN_IDS labels

Design
------
* Five fields are closed vocabularies in the training labels
  (visa_class 5, fee_status 4, declared_purpose 10, species_code 12, home_world 13).
  They are mined from TRAIN_IDS labels ONLY and frozen into vocab_train.json.
  applicant_name (979 values) and sponsor_id (864) are open -> never vocabulary-snapped.
* Similarity runs on an OCR-canonical key: casefold, collapse OCR confusion classes
  (0/O/o, 1/l/I/i, 5/S, 8/B, 2/Z, rn->m) and drop all separators/punctuation.
  Dropping separators makes the common "merged token" error
  ('reactormaintenance', 'EuropaStation') an *exact* key match at zero edit distance.
* A snap only happens when the match is close AND unambiguous:
    - normalized edit-distance similarity >= SIM_THRESHOLD, and
    - it beats the runner-up vocabulary entry by >= MARGIN, or
    - the raw key is a unique substring of exactly one vocabulary key
      (>= MIN_CONTAIN_LEN chars), which recovers truncated OCR ('ORION', 'Sirius').
  Otherwise the RAW VALUE IS RETURNED UNCHANGED: the private test set will contain
  species/worlds absent from train, and hard-snapping those to a wrong known value is
  strictly worse than passing them through.
* sponsor_id is pattern-repaired to the SPN-#### shape, never vocabulary-snapped.
* Only visible-OCR-derived strings should ever be passed in. Nothing here reads
  hidden_text; it cannot invent a value, it can only repair one already extracted.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VOCAB_PATH = os.environ.get("MIB_VOCAB") or os.path.join(HERE, "vocab_train.json")

CLOSED_FIELDS = ("visa_class", "fee_status", "declared_purpose", "species_code", "home_world")
OPEN_FIELDS = ("applicant_name", "sponsor_id")

# Values that mean "we did not find anything" - never snap these, and never let
# them be treated as evidence.
NULLISH = {"", "unknown", "none", "null", "n/a", "na", "-", "--"}

# ----------------------------------------------------------------- thresholds
# Tuned on TRAIN_IDS only (see __main__ sweep). See module docstring for rationale.
SIM_THRESHOLD = {
    # short, dense codes: XW-1 vs XW-2 are 0.75 similar, so the bar must sit above
    # that or a genuinely new class ('XW-3') would be snapped onto a known one.
    "visa_class": 0.85,
    "fee_status": 0.80,
    # long multi-word strings: OCR noise is proportionally smaller, so a tighter
    # relative bar is still safe and the runner-up is far away.
    "declared_purpose": 0.72,
    "species_code": 0.72,
    "home_world": 0.72,
}
DEFAULT_SIM_THRESHOLD = 0.80
# best must beat runner-up by this much (absolute similarity) to be unambiguous
MARGIN = 0.06
# Truncation recovery ('ORION' -> 'ORION_GRAYS') only applies to long vocabulary
# entries. For short codes, a truncation is indistinguishable from a *different*
# value - e.g. 'paid' is contained in 'unpaid', and snapping it there would invert
# the meaning. Leave-one-value-out on TRAIN shows the false-snap rate for
# fee_status drops 0.687 -> 0.000 once MIN_TARGET_LEN is enforced.
MIN_CONTAIN_LEN = 4
MIN_TARGET_LEN = 8

# ----------------------------------------------------------------- canonical key
_CONFUSION = {}
for _grp, _rep in (("oO0", "0"), ("iIlL1|!", "1"), ("sS5", "5"),
                   ("bB8", "8"), ("zZ2", "2")):
    for _ch in _grp:
        _CONFUSION[_ch] = _rep


def canon(s: str) -> str:
    """OCR-tolerant comparison key: confusion classes folded, separators removed."""
    s = str(s or "").casefold()
    s = "".join(_CONFUSION.get(ch, ch) for ch in s)
    s = s.replace("rn", "m")
    return re.sub(r"[^a-z0-9]", "", s)


def normalize(value: str) -> str:
    """evaluate.py semantics: collapse whitespace + casefold."""
    return " ".join(str(value or "").strip().split()).casefold()


# ----------------------------------------------------------------- edit distance
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    return 1.0 - levenshtein(a, b) / m


# ----------------------------------------------------------------- vocabulary
_VOCAB = None


def build_vocab(force: bool = False, write: bool = False, mine: bool = False):
    """Mine closed-field vocabularies from TRAIN_IDS labels ONLY.

    Read path (the only one inference takes): load the frozen vocab_train.json.
    Mining requires the train labels and is DEV-TIME ONLY, so it never happens
    implicitly and it NEVER writes unless the caller explicitly asks (`write=True`),
    which keeps this safe on a read-only container rootfs.
    """
    if os.path.exists(VOCAB_PATH) and not force:
        with open(VOCAB_PATH) as f:
            return json.load(f)
    if not mine:
        # Inference must NEVER reach the training labels. Without the frozen
        # artifact we raise, and get_vocab() degrades to an empty vocabulary.
        raise FileNotFoundError(
            f"vocab artifact missing at {VOCAB_PATH}; pass mine=True to re-mine (dev only)")
    import common
    labels = common.load_labels()
    train_ids, _dev_ids = common.split_ids(labels)      # dev is never touched
    vocab = {}
    for field in CLOSED_FIELDS:
        counts = {}
        for cid in train_ids:
            v = (labels[cid].get(field) or "").strip()
            if not v or normalize(v) in NULLISH:
                continue
            counts[v] = counts.get(v, 0) + 1
        vocab[field] = sorted(counts, key=lambda v: (-counts[v], v))
    if write:
        with open(VOCAB_PATH, "w") as f:
            json.dump(vocab, f, indent=1, sort_keys=True)
    return vocab


def get_vocab():
    """Frozen vocabulary, or an EMPTY one if the artifact is missing/unreadable.

    An empty vocab is the documented safe degradation: snap() finds no candidate
    and passes every value through unchanged. Losing OCR correction is a few
    points; raising here would zero out every case in the run.
    """
    global _VOCAB
    if _VOCAB is None:
        try:
            raw = build_vocab()
            if not isinstance(raw, dict):
                raise TypeError("vocab artifact is not a dict")
            _VOCAB = {k: [(v, canon(v)) for v in vs]
                      for k, vs in raw.items() if isinstance(vs, list)}
        except Exception as e:                     # missing / unreadable / malformed
            print(f"[vocab] WARNING: no usable vocabulary ({type(e).__name__}: {e}); "
                  f"snapping disabled, raw OCR values pass through unchanged",
                  file=sys.stderr)
            _VOCAB = {}
    return _VOCAB


# ----------------------------------------------------------------- sponsor_id
_SPN_DIGIT = {"o": "0", "O": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1",
              "|": "1", "!": "1", "L": "1", "Z": "2", "z": "2", "S": "5", "s": "5",
              "B": "8", "b": "6", "G": "6", "g": "9", "q": "9", "T": "7", "?": "7",
              "A": "4", "e": "8"}


def repair_sponsor_id(raw: str):
    """Coerce a noisy string to the SPN-#### shape. Returns None if it cannot be."""
    s = str(raw or "")
    if not s.strip():
        return None
    m = re.search(r"SPN[\s\-_.:]*([A-Za-z0-9|!?]{4})", s, re.I)
    if not m:
        # OCR may mangle the 'SPN' prefix itself (5PN, 3PN, SPh...). Accept a
        # 3-char token whose canonical form is 'spn' followed by 4 chars.
        m = re.search(r"\b([A-Za-z0-9$5]{3})[\s\-_.:]*([A-Za-z0-9|!?]{4})\b", s)
        if not m or canon(m.group(1)) != canon("SPN"):
            return None
        digits = m.group(2)
    else:
        digits = m.group(1)
    fixed = "".join(_SPN_DIGIT.get(ch, ch) for ch in digits)
    if not re.fullmatch(r"\d{4}", fixed):
        return None
    return "SPN-" + fixed


# ----------------------------------------------------------------- name repair
_ROMAN_ISH = re.compile(r"[A-Z][a-z]+")


def repair_name(raw: str) -> str:
    """Open-vocabulary repairs for applicant_name: split merged CamelCase tokens and
    fix a leading lowercase 'l' that should be a capital 'I'. No vocabulary is used,
    so unseen names are handled the same as seen ones."""
    s = " ".join(str(raw or "").split())
    if not s or normalize(s) in NULLISH:
        return raw
    out = []
    for tok in s.split(" "):
        # split CamelCase merges: 'MiraixVeerix' -> 'Miraix Veerix'
        if len(tok) > 3 and re.fullmatch(r"[A-Za-z]+", tok):
            tok = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok)
        out.extend(tok.split(" "))
    fixed = []
    for tok in out:
        # 'lxodane' -> 'Ixodane': a lowercase-initial alphabetic token starting with
        # 'l' in an otherwise title-cased name is the classic l/I confusion.
        if len(tok) > 2 and tok[0] == "l" and tok[1:].islower() and tok.isalpha():
            tok = "I" + tok[1:]
        fixed.append(tok)
    return " ".join(t for t in fixed if t)


# ----------------------------------------------------------------- core snap
def match_detail(field: str, raw_text: str, vocab=None):
    """Return (value, reason, best_sim, runner_up_sim) without deciding.

    `vocab` overrides the mined vocabulary (used by the leave-one-value-out
    threshold sweep to simulate a value the private set has but train does not).
    """
    raw = " ".join(str(raw_text or "").split())
    vocab = vocab if vocab is not None else get_vocab().get(field)
    if not vocab or normalize(raw) in NULLISH:
        return raw, "no-op", 0.0, 0.0
    key = canon(raw)
    if not key:
        return raw, "no-op", 0.0, 0.0

    scored = sorted(((similarity(key, vk), v, vk) for v, vk in vocab), reverse=True)
    best_sim, best_val, best_key = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0

    if best_sim >= 1.0:
        return best_val, "exact", best_sim, runner

    thr = SIM_THRESHOLD.get(field, DEFAULT_SIM_THRESHOLD)
    if best_sim >= thr and (best_sim - runner) >= MARGIN:
        return best_val, "similar", best_sim, runner

    # truncation recovery: raw key is a substring of exactly one vocabulary key
    if len(key) >= MIN_CONTAIN_LEN:
        hits = [v for v, vk in vocab if key in vk and len(vk) >= MIN_TARGET_LEN]
        if len(hits) == 1:
            return hits[0], "contained", best_sim, runner

    return raw, "kept-raw", best_sim, runner


def snap(field: str, raw_text: str) -> str:
    """Snap a noisy OCR string for `field` to the nearest TRAIN vocabulary entry.

    Returns raw_text unchanged when the field is open-vocabulary, when the value is
    nullish, or when no vocabulary entry is close enough to be trusted.
    """
    if field == "sponsor_id":
        return repair_sponsor_id(raw_text) or raw_text
    if field == "applicant_name":
        return repair_name(raw_text)
    if field not in CLOSED_FIELDS:
        return raw_text
    return match_detail(field, raw_text)[0]


def find_in_text(field: str, text: str):
    """Rescue path: look for a vocabulary entry inside TRUSTED visible OCR text.

    Used only when extraction produced nothing at all. Never call this with
    hidden_text - it is the injection surface and may not supply values.
    """
    vocab = get_vocab().get(field)
    if not vocab or not text:
        return None
    key = canon(text)
    hits = [(v, vk) for v, vk in vocab if len(vk) >= 4 and vk in key]
    if not hits:
        return None
    # nested vocabulary entries ('paid' inside 'unpaid'): the longer, more specific
    # entry wins; only bail out when two genuinely different entries both appear.
    longest = max(hits, key=lambda h: len(h[1]))
    if all(h[1] in longest[1] for h in hits):
        return longest[0]
    return None


def find_sponsor_in_text(text: str):
    """Scan TRUSTED visible OCR text for a repairable SPN-#### token."""
    if not text:
        return None
    cands = []
    for m in re.finditer(r"[A-Za-z0-9$|!?]{2,4}[\s\-_.:]*[A-Za-z0-9|!?]{4}", str(text)):
        got = repair_sponsor_id(m.group(0))
        if got:
            cands.append(got)
    uniq = set(cands)
    return cands[0] if len(uniq) == 1 else None


def snap_fields(fields: dict, visible_text: str = None) -> dict:
    """Snap every field of an extracted record. `visible_text` (TRUSTED OCR only)
    enables the rescue path for fields extraction left as unknown."""
    out = dict(fields)
    for f in CLOSED_FIELDS:
        if f not in out:
            continue
        v = out[f]
        if normalize(v) in NULLISH:
            if visible_text:
                got = find_in_text(f, visible_text)
                if got:
                    out[f] = got
            continue
        val, reason, _b, _r = match_detail(f, v)
        out[f] = val
        if reason == "kept-raw" and visible_text:
            # extraction found *something* but it matches no known value. It may be a
            # legitimately new value (keep it) or a misgrab of a neighbouring cell.
            # Only override when the document text itself names exactly one vocab entry.
            got = find_in_text(f, visible_text)
            if got:
                out[f] = got
    if "sponsor_id" in out:
        sid = repair_sponsor_id(out["sponsor_id"])
        if sid is None or sid == "SPN-0000":
            sid = (find_sponsor_in_text(visible_text) if visible_text else None) or sid
        out["sponsor_id"] = sid or out["sponsor_id"]
    if "applicant_name" in out:
        out["applicant_name"] = repair_name(out["applicant_name"])
    return out


# ================================================================== self-test
def _selftest():
    import common
    import baseline

    build_vocab(force=True, write=True, mine=True)   # dev-time re-mine: the ONLY writer
    get_vocab()
    cache = common.load_cache()
    labels = common.load_labels()
    train_ids, dev_ids = common.split_ids([c for c in cache if c in labels])
    print(f"cache={len(cache)}  train={len(train_ids)}  dev={len(dev_ids)}")
    print("vocab sizes:", {k: len(v) for k, v in get_vocab().items()})

    fields_of = {cid: baseline.extract_rec(cache[cid]) for cid in train_ids + dev_ids}
    vis_of = {cid: common.visible_text(cache[cid]) for cid in train_ids + dev_ids}

    def acc(ids, field, fn):
        ok = 0
        for cid in ids:
            if normalize(fn(cid, field)) == normalize(labels[cid][field]):
                ok += 1
        return ok / max(len(ids), 1)

    # ---- threshold sweep on TRAIN ONLY
    # gain  = train accuracy of snap() over the baseline extraction
    # harm  = leave-one-value-out: the case's true value is deleted from the
    #         vocabulary (simulating a species/world the private set has and train
    #         does not) and we count how often snap() still fires -> a wrong snap
    #         where passing the raw value through would have been correct-or-neutral.
    print("\n--- threshold sweep (TRAIN only) ---")
    saved = dict(SIM_THRESHOLD)
    full = get_vocab()
    hdr = " ".join(f"{f[:8]:>16}" for f in CLOSED_FIELDS)
    print(f"{'thr':>5} " + hdr + "     (acc / loo-false-snap-rate)")
    for thr in (0.30, 0.40, 0.50, 0.60, 0.66, 0.72, 0.80, 0.85, 0.90, 1.01):
        for f in CLOSED_FIELDS:
            SIM_THRESHOLD[f] = thr
        cells = []
        for f in CLOSED_FIELDS:
            a = acc(train_ids, f, lambda c, fl: snap(fl, fields_of[c][fl]))
            bad = n = 0
            for cid in train_ids:
                raw = fields_of[cid][f]
                if normalize(raw) in NULLISH:
                    continue
                truth = labels[cid][f]
                reduced = [(v, vk) for v, vk in full[f] if normalize(v) != normalize(truth)]
                if len(reduced) == len(full[f]):
                    continue
                n += 1
                val, reason, _, _ = match_detail(f, raw, vocab=reduced)
                if reason != "kept-raw" and normalize(val) != normalize(raw):
                    bad += 1
            cells.append(f"{a:.3f}/{(bad / n if n else 0):.3f}")
        print(f"{thr:>5.2f} " + " ".join(f"{c:>16}" for c in cells))
    SIM_THRESHOLD.clear()
    SIM_THRESHOLD.update(saved)
    print("chosen thresholds:", SIM_THRESHOLD, "MARGIN=", MARGIN)

    # ---- DEV metrics: before vs after
    def report(ids, tag):
        print(f"\n--- per-field exact match on {tag} (n={len(ids)}) ---")
        print(f"{'field':>17} {'before':>8} {'snap':>8} {'+rescue':>8}")
        snapped = {c: snap_fields(fields_of[c]) for c in ids}
        rescued = {c: snap_fields(fields_of[c], vis_of[c]) for c in ids}
        for f in list(CLOSED_FIELDS) + list(OPEN_FIELDS):
            b = acc(ids, f, lambda c, fl: fields_of[c][fl])
            s = acc(ids, f, lambda c, fl: snapped[c][fl])
            r = acc(ids, f, lambda c, fl: rescued[c][fl])
            print(f"{f:>17} {b:>8.3f} {s:>8.3f} {r:>8.3f}   "
                  f"delta_snap={s - b:+.3f} delta_rescue={r - b:+.3f}")

    report(train_ids, "TRAIN (fit set - sanity only)")
    report(dev_ids, "DEV (held out)")

    # ---- damage check: how often does snapping CHANGE an already-correct value?
    broke = kept = fixed = 0
    for cid in dev_ids:
        for f in CLOSED_FIELDS:
            b, s, t = fields_of[cid][f], snap(f, fields_of[cid][f]), labels[cid][f]
            if normalize(b) == normalize(s):
                continue
            if normalize(b) == normalize(t):
                broke += 1
            elif normalize(s) == normalize(t):
                fixed += 1
            else:
                kept += 1
    print(f"\nDEV snap edits: fixed={fixed}  broke={broke}  neutral={kept}")

    import collections
    reasons = collections.Counter()
    for cid in dev_ids:
        for f in CLOSED_FIELDS:
            raw = fields_of[cid][f]
            if normalize(raw) in NULLISH:
                reasons[(f, "nullish")] += 1
                continue
            reasons[(f, match_detail(f, raw)[1])] += 1
    print("DEV match-path breakdown:")
    for f in CLOSED_FIELDS:
        print(f"  {f:>17}: " + " ".join(f"{r}={c}" for (ff, r), c in sorted(reasons.items()) if ff == f))

    # ---- end-to-end score with the REAL evaluate.py (dev only)
    base_rows = [baseline.row_for(c, fields_of[c]) for c in dev_ids]
    snap_rows = [baseline.row_for(c, snap_fields(fields_of[c], vis_of[c])) for c in dev_ids]
    print("\nDEV end-to-end (scripts/evaluate.py):")
    print("  before:", common.summarize(common.score_rows(base_rows, "vocab_before")))
    print("  after :", common.summarize(common.score_rows(snap_rows, "vocab_after")))

    # ---- unit checks for the no-match fallback
    for f, junk in (("species_code", "KLINGON_WARRIOR"), ("home_world", "Qo'noS Prime"),
                    ("visa_class", "XW-9"), ("declared_purpose", "hyperspace tourism")):
        got = snap(f, junk)
        print(f"unseen {f:>17}: {junk!r} -> {got!r} {'OK' if got == junk else 'SNAPPED!'}")
    print("sponsor repair:", [repair_sponsor_id(x) for x in
                              ["SPN-4732", "SPN 4732", "SPNA732", "5PN-O132", "garbage", ""]])


if __name__ == "__main__":
    _selftest()
