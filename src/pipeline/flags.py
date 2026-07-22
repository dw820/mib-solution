"""Risk-flag detection for the MIB Doc Challenge (distant supervision on TRAIN_IDS).

Public API
----------
    detect_flags(rec)          -> set[str]   (subset of VOCAB; empty set == 'none')
    risk_flags_string(rec)     -> str        ('none' or 'a|b|c', alphabetically sorted)
    detect_flags_detail(rec)   -> (set[str], evidence dict)
    flags_confidence(rec)      -> float in [0,1]  (how much the packet actually told us)

`rec` is a record from common.load_cache().  Only visible OCR is read
(common.doc_lines / page lines).  hidden_text is never touched.

What the training data actually looks like
------------------------------------------
Mining TRAIN_IDS (per-token lift per flag, then reading the winners) showed the
packets state their risk posture in exactly three stereotyped VISIBLE places and
nowhere else:

  1. FORM B-13 Biometric Scan Slip -> "Observed flags: <list>"  (or "...: none").
     This is the authoritative enumeration; whenever it is legible it agreed
     with the label essentially always.
  2. Manual Adjudicator Note reasons:
        "Reason: Disqualifying risk flag: <name>."
        "Reason: Review-only risk flag present: <name>."
        "Prior denial stamp - rescinded. Route to human review."
  3. Nothing at all.  Roughly half the packets carry neither a slip nor a note;
     for those the flags are simply not in visible evidence.  We answer 'none'
     rather than guess (see "What was tried and rejected").

The real difficulty is OCR robustness, not vocabulary.  The anchor and the flag
names arrive shredded:

    'Cboanved flgec nom'             -> Observed flags: none
    'Obsarved fiags:legbieotics'     -> illegible_biometrics
    'Ybserved ffoga:Bagle_hlcmtics'  -> illegible_biometrics
    'Obsevedfgs:active_wntl_bmic'    -> active_warrant (+ illegible_biometrics)
    'Reason:Disqualifying riskflag:planetary_er nbargo'

So both the anchor and the flag names are matched with a sliding-window fuzzy
matcher (difflib ratio over an alphanumeric-squashed string), with greedy
removal so several flags can be pulled out of one mangled field.  Because the
match is to the *canonical vocabulary*, an unseen layout or a flag never seen in
training is still recognised as long as the packet spells it.

What was tried and rejected (all measured on TRAIN_IDS)
------------------------------------------------------
  * "Registry Status: EMBARGO_REVIEW" and "Reason: Embargo home world: X" ->
    8 false positives for planetary_embargo.  These mark an embargoed *home
    world*, a separate denial reason, not the risk flag.  Dropped.
  * The "SAMPLE DENIAL" watermark and bare "DENIED" stamps -> documented traps
    (FIELD_MANUAL "Known Document Traps"); never used as evidence.  rescinded
    _denial requires the literal 'rescind' stem, not a denial stamp.
  * A HistGradientBoostingClassifier over damage/quality/consistency features
    for the evidence-free packets: out-of-fold AUC was 0.36-0.72 (chance) for
    every flag, so the fallback would have been noise.  Dropped; the honest
    answer for those packets is 'none'.
  * Home-world / species vocabularies correlated with flags -> pure
    memorisation of the public set; would not transfer.  Dropped.

Nothing in this module is keyed to a case_id, and no threshold was tuned on DEV.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common  # noqa: E402

DISQUALIFYING = ("memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red")
REVIEW_ONLY = ("identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial")
VOCAB = tuple(sorted(DISQUALIFYING + REVIEW_ONLY))

_SQ_VOCAB = {f: re.sub(r"[^a-z0-9]", "", f) for f in VOCAB}

# ---- thresholds, all selected on TRAIN_IDS only ---------------------------- #
ANCHOR_THR = 0.60       # fuzzy "observed flags" anchor on the slip
FLAG_THR = 0.62         # fuzzy flag-name match in the slip's value field
FLAG_THR_LOW = 0.55     # second pass when the first pass found nothing
NONE_THR = 0.65         # value fuzzy-matches the literal 'none' -> no flags
NOTE_FLAG_THR = 0.70    # stricter inside adjudicator-note prose (more distractors)
NAME_CONF = 0.97        # OCR confidence required to trust a name comparison
NAME_RATIO = 0.60       # below this similarity two trusted names are "different"
# ---------------------------------------------------------------------------- #

_ANCHOR_OBS = "observedflags"
_PAGE_FURNITURE = re.compile(
    r"scanimage|scantab|packet|synthetic|casework|archive|copyartifact|redacted"
    r"|sampledenial|sampledenia|intake|filed|image"
)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _squash(s: str) -> str:
    """lowercase, alphanumerics only -- OCR both invents and eats whitespace."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _page_lines(rec):
    for p in common.pages_of(rec):
        yield p, [l["t"] for l in common.page_lines(p)]


def _best_window(pat: str, s: str, lo: float = 0.55, hi: float = 1.5):
    """(ratio, start, end) for the substring of `s` most similar to `pat`."""
    n = len(pat)
    if not s or not pat:
        return 0.0, 0, 0
    best = (0.0, 0, 0)
    w_lo = max(3, int(n * lo))
    for w in range(w_lo, int(n * hi) + 1):
        if w > len(s):
            if best[0] == 0.0:
                best = (difflib.SequenceMatcher(None, pat, s).ratio(), 0, len(s))
            break
        for i in range(0, len(s) - w + 1):
            r = difflib.SequenceMatcher(None, pat, s[i:i + w]).ratio()
            if r > best[0]:
                best = (r, i, i + w)
    return best


def _match_flags(value: str, thr: float):
    """Greedily pull canonical flag names out of a (possibly shredded) field."""
    s = _squash(value)
    if len(s) < 4:
        return set()
    got = set()
    for _ in range(4):                       # at most 4 flags in one field
        best = (0.0, None, 0, 0)
        for flag, sqf in _SQ_VOCAB.items():
            if flag in got:
                continue
            r, a, b = _best_window(sqf, s)
            if r > best[0]:
                best = (r, flag, a, b)
        if best[1] is None or best[0] < thr:
            break
        got.add(best[1])
        s = s[:best[2]] + s[best[3]:]        # consume the matched span, look again
        if len(s) < 5:
            break
    return got


# --------------------------------------------------------------------------- #
# source 1: FORM B-13 biometric scan slip, "Observed flags: ..."
# --------------------------------------------------------------------------- #
def _is_slip_page(page_sq: str) -> bool:
    if "biometric" in page_sq or "scanimage" in page_sq or "speciesmatch" in page_sq:
        return True
    if re.search(r"formb[-_. ]?\d{1,2}", page_sq):
        return True
    return _best_window("biometricscanslip", page_sq[:160])[0] >= 0.62


def _slip_values(rec):
    """Squashed value strings that follow an 'Observed flags' anchor on a slip."""
    vals = []
    for _p, lines in _page_lines(rec):
        psq = _squash(" ".join(lines))
        if not _is_slip_page(psq):
            continue
        for t in lines:
            s = _squash(t)
            if len(s) < 6 or len(s) > 90:
                continue
            r, _a, b = _best_window(_ANCHOR_OBS, s[:len(_ANCHOR_OBS) + 6])
            if r >= ANCHOR_THR:
                vals.append(_PAGE_FURNITURE.split(s[b:])[0])
    return vals


def _observed_flags(rec):
    """(flags, anchor_found, all_values_resolved)."""
    vals = _slip_values(rec)
    flags = set()
    unresolved = False
    for v in vals:
        if not v:
            unresolved = True
            continue
        if _best_window("none", v)[0] >= NONE_THR:
            continue                          # the slip explicitly says 'none'
        f = _match_flags(v, FLAG_THR) or _match_flags(v, FLAG_THR_LOW)
        if f:
            flags |= f
        else:
            unresolved = True
    return flags, bool(vals), not unresolved


# --------------------------------------------------------------------------- #
# source 2: manual adjudicator note reasons
# --------------------------------------------------------------------------- #
_A_RISKFLAG = re.compile(r"riskflag[a-z]{0,10}?(?:present)?")
_A_DISQ = re.compile(r"disqualif[a-z]{0,12}")
_A_RESCIND = re.compile(r"priordenialstamp")


def _note_flags(rec):
    flags, found = set(), False
    for _p, lines in _page_lines(rec):
        sq = _squash(" ".join(lines))
        for rx in (_A_RISKFLAG, _A_DISQ):
            for m in rx.finditer(sq):
                w = _PAGE_FURNITURE.split(sq[m.end(): m.end() + 46])[0]
                w = re.split(r"finding|reason", w)[0]
                f = _match_flags(w, NOTE_FLAG_THR)
                if f:
                    found = True
                    flags |= f
        # a denial stamp that a later note rescinds -> rescinded_denial.
        # (A bare DENIED stamp or a "sample denial" watermark is NOT evidence.)
        if _A_RESCIND.search(sq) and "rescind" in sq:
            found = True
            flags.add("rescinded_denial")
    return flags, found


# --------------------------------------------------------------------------- #
# source 3: cross-document identity contradiction
# --------------------------------------------------------------------------- #
_NAME_KEYS = ("registryname", "applicant", "applicantname")


def _trusted_names(rec):
    """(registry_name, intake_name) -- only names OCR'd with high confidence, so a
    difference means the documents really disagree, not that the scan is dirty."""
    reg = intake = ""
    for p, _lines in _page_lines(rec):
        lines = common.page_lines(p)
        psq = _squash(" ".join(l["t"] for l in lines))
        is_reg = "planetaryregistry" in psq
        is_int = "workauthorizationintake" in psq
        if not (is_reg or is_int):
            continue
        for i, l in enumerate(lines):
            s = _squash(l["t"])
            if not any(s.startswith(k) for k in _NAME_KEYS):
                continue
            conf = float(l.get("s", 0.0))
            if ":" in l["t"]:
                val = l["t"].split(":")[-1]
            elif i + 1 < len(lines):
                val = lines[i + 1]["t"]
                conf = min(conf, float(lines[i + 1].get("s", 0.0)))
            else:
                continue
            v = re.sub(r"[^a-z]", "", val.lower())
            if len(v) > 5 and conf >= NAME_CONF:
                if is_reg and not reg:
                    reg = v
                elif is_int and not intake:
                    intake = v
    return reg, intake


def _identity_conflict(rec) -> bool:
    reg, intake = _trusted_names(rec)
    if not reg or not intake:
        return False
    return difflib.SequenceMatcher(None, reg, intake).ratio() < NAME_RATIO


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def detect_flags_detail(rec):
    obs, obs_found, obs_clean = _observed_flags(rec)
    note, note_found = _note_flags(rec)

    # The slip is the authoritative enumeration (FIELD_MANUAL rank 3). When it is
    # present the note may only ADD a flag the slip could not resolve.
    flags = (obs | note) if obs_found else set(note)

    # A trusted, unambiguous name contradiction is direct visible evidence, but
    # it must not override a slip that legibly enumerated every flag.
    id_conf = _identity_conflict(rec)
    if id_conf and not (obs_found and obs_clean):
        flags.add("identity_conflict")

    ev = {
        "slip_anchor": obs_found,
        "slip_resolved": obs_clean,
        "slip_flags": sorted(obs),
        "note_flags": sorted(note),
        "name_conflict": id_conf,
        "anchor": obs_found or note_found,
    }
    return flags, ev


def detect_flags(rec):
    """rec -> set of risk-flag strings. Empty set means 'none'."""
    return detect_flags_detail(rec)[0]


def risk_flags_string(rec):
    """rec -> the pipe-delimited value evaluate.py scores ('none' when empty)."""
    f = detect_flags(rec)
    return "|".join(sorted(f)) if f else "none"


def flags_confidence(rec):
    """How much the packet actually told us about its risk posture.
    1.00 legible slip enumeration | 0.75 slip present but partly unreadable, or
    only an adjudicator note | 0.45 no visible risk evidence at all."""
    _f, ev = detect_flags_detail(rec)
    if ev["slip_anchor"] and ev["slip_resolved"]:
        return 1.0
    if ev["anchor"]:
        return 0.75
    return 0.45


# --------------------------------------------------------------------------- #
# evaluation / self-test
# --------------------------------------------------------------------------- #
def _true_flags(row):
    v = (row.get("risk_flags") or "none").strip()
    return set() if v in ("", "none") else set(v.split("|"))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _report(ids, cache, labels, title):
    import collections
    per = collections.defaultdict(collections.Counter)
    dq = collections.Counter()
    cov = collections.Counter()
    exact = 0
    for cid in ids:
        pred, ev = detect_flags_detail(cache[cid])
        true = _true_flags(labels[cid])
        exact += pred == true
        if ev["anchor"]:
            cov["anchored"] += 1
            cov["anchored_exact"] += pred == true
        else:
            cov["blind"] += 1
            cov["blind_exact"] += pred == true
            cov["blind_but_flagged"] += bool(true)
        for f in VOCAB:
            per[f]["tp" if (f in true and f in pred) else
                   "fn" if f in true else
                   "fp" if f in pred else "tn"] += 1
        pd_, td = bool(pred & set(DISQUALIFYING)), bool(true & set(DISQUALIFYING))
        dq["tp"] += pd_ and td
        dq["fp"] += pd_ and not td
        dq["fn"] += (not pd_) and td
    print(f"\n=== {title}  (n={len(ids)}) ===")
    print(f"{'flag':24s} {'sup':>4s} {'tp':>4s} {'fp':>4s} {'fn':>4s} "
          f"{'prec':>6s} {'rec':>6s} {'F1':>6s}")
    for f in VOCAB:
        c = per[f]
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        print(f"{f:24s} {c['tp']+c['fn']:4d} {c['tp']:4d} {c['fp']:4d} {c['fn']:4d} "
              f"{p:6.3f} {r:6.3f} {f1:6.3f}")
    tp = sum(per[f]["tp"] for f in VOCAB)
    fp = sum(per[f]["fp"] for f in VOCAB)
    fn = sum(per[f]["fn"] for f in VOCAB)
    p, r, f1 = _prf(tp, fp, fn)
    print(f"{'MICRO-AVG':24s} {tp+fn:4d} {tp:4d} {fp:4d} {fn:4d} {p:6.3f} {r:6.3f} {f1:6.3f}")
    p, r, f1 = _prf(dq["tp"], dq["fp"], dq["fn"])
    print(f"{'ANY-DISQUALIFYING':24s} {dq['tp']+dq['fn']:4d} {dq['tp']:4d} {dq['fp']:4d} "
          f"{dq['fn']:4d} {p:6.3f} {r:6.3f} {f1:6.3f}")
    print(f"EXACT-MATCH risk_flags string : {exact}/{len(ids)} = {exact/len(ids):.4f}")
    if cov["anchored"]:
        print(f"  visible risk evidence   : {cov['anchored']:4d} cases "
              f"({cov['anchored']/len(ids):.3f}) -> exact "
              f"{cov['anchored_exact']}/{cov['anchored']} = "
              f"{cov['anchored_exact']/cov['anchored']:.4f}")
    if cov["blind"]:
        print(f"  no visible risk evidence: {cov['blind']:4d} cases "
              f"({cov['blind']/len(ids):.3f}) -> exact "
              f"{cov['blind_exact']}/{cov['blind']} = "
              f"{cov['blind_exact']/cov['blind']:.4f}   "
              f"({cov['blind_but_flagged']} of them are truly flagged but the "
              f"packet never shows it)")
    return exact / len(ids)


def _baseline(ids, labels, title):
    exact = sum(1 for c in ids if not _true_flags(labels[c]))
    print(f"\n--- BASELINE (always 'none'), {title}: exact "
          f"{exact}/{len(ids)} = {exact/len(ids):.4f}; "
          f"disqualifying recall 0.000 ---")
    return exact / len(ids)


def _main():
    cache = common.load_cache()
    labels = common.load_labels()
    ids = sorted(i for i in cache if i in labels)
    train_ids, dev_ids = common.split_ids(ids)
    print(f"cache={len(cache)} usable={len(ids)} train={len(train_ids)} dev={len(dev_ids)}")
    print(f"thresholds fit on TRAIN only: anchor={ANCHOR_THR} flag={FLAG_THR}/"
          f"{FLAG_THR_LOW} none={NONE_THR} note={NOTE_FLAG_THR} "
          f"name={NAME_CONF}/{NAME_RATIO}")
    _report(train_ids, cache, labels, "TRAIN (thresholds tuned here)")
    b = _baseline(dev_ids, labels, "DEV")
    d = _report(dev_ids, cache, labels, "DEV (held out)")
    print(f"\nDEV exact-match risk_flags: baseline {b:.4f} -> detector {d:.4f} "
          f"({d - b:+.4f})")


if __name__ == "__main__":
    _main()
