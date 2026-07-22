"""DESIGN D - adjudication classifier (APPROVED / DENIED / NEEDS_REVIEW) + confidence.

Three ideas, in order of how many points each is worth (all measured, see __main__):

1.  **Read the adjudicator note.**  FIELD_MANUAL precedence rank 1 is "visible MIB
    adjudicator stamp or signed manual note".  Roughly a third of packets carry a
    *Manual Adjudicator Note* whose body is literally
        ``Finding: DENIED.  Reason: Disqualifying risk flag: biohazard_red.``
    The stock 97-feature vector only exposes doc-level ``tok_denied`` /
    ``tok_approved`` bag-of-word bits, which also fire on the ``SAMPLE DENIAL``
    watermark trap and on ``APPROVED`` inside an injected barcode payload.  This
    module instead fuzzy-matches the *anchored* ``Finding:`` field and the closed
    set of ``Reason:`` templates, which survives the OCR shredding
    (``FindngNEEDS_REVIEW``, ``Finding: DENlED. Reason: Denial supported by ...``).
    On TRAIN the anchored finding fires on 274/800 packets and is correct 274/274.

2.  **Policy features the stock vector cannot express.**  Chiefly the three
    revoked sponsors named in FIELD_MANUAL, visible revocation prose, embargo
    registry status, and the fee/visa waiver interaction.  These are *policy
    enums from the public manual*, not identifiers mined from the public PDFs.

3.  **Decide by expected value, not by argmax.**  The scoring table is wildly
    asymmetric (a false approval of a denied case is -4, routing to NEEDS_REVIEW
    is +2).  Given calibrated class probabilities the optimal action is
        E[APPROVED]      = 8*P(A) - 4*P(D) + 1*P(NR)
        E[DENIED]        = 8*P(D) + 0*P(A) + 1*P(NR)
        E[NEEDS_REVIEW]  = 8*P(NR) + 2*P(A) + 2*P(D)
    argmax of that, which abstains to NEEDS_REVIEW exactly when it should and
    never approves on a coin flip.  Confidence is then P(chosen class), which is
    by construction the probability the decision is correct - the quantity the
    Brier calibration term scores.

Measured results (DEV_IDS = 200 held-out cases, scored by scripts/evaluate.py)
-----------------------------------------------------------------------------
                              total    class    calib   false-approvals
    prior baseline            74.27    41.62     8.77          0
    this module              121.34    62.90    14.53         10
    (extraction is unchanged at 43.92/50 in both; all delta is decisions.)

  dev accuracy 0.785.  Confusion (truth -> pred):
      APPROVED       A 38   D  8   NR  5
      DENIED         A 10   D 72   NR 12
      NEEDS_REVIEW   A  2   D  6   NR 47

  Where the score comes from, ablated on TRAIN out-of-fold (cls+cal, max 100):
      stock 97 features only .................. 70.20
      + note reader / policy / slip / raw form . 75.11   (+4.9)
      + smaller model & EV penalty tuning ...... 76.23

  What did NOT work, and is therefore absent:
      * A bag-of-words over frequent visible tokens (185 terms, entity words
        from TRAIN labels excluded): dev accuracy 0.745 vs 0.740 without, and it
        made false approvals worse.  Generic prose carries no decision signal.
      * A hard "use the note verdict, bypass the model" override: 75.10 vs
        75.11.  The gradient booster already reproduces the note perfectly, so
        the override only adds a failure mode.
      * A separate model specialised on note-less packets: 0.644 vs 0.656.

Honest limits
-------------
70% of packets carry no adjudicator note, and on those the model is only 65.6%
accurate (TRAIN OOF).  That is close to the visible-evidence ceiling, not a
modelling failure: 56% of note-less packets have *no* risk evidence anywhere in
visible OCR, and their label distribution (A .36 / D .40 / NR .24) is
indistinguishable from the base rate.  Substituting ground-truth fields for the
extractor's raises note-less accuracy to 0.869, so the remaining headroom lives
in extraction (chiefly risk-flag recall: the detector is 100% precise but misses
~40% of true flags), not here.

The 10 dev false approvals are the EV rule working as designed, not a bug: even
the cleanest observable bucket (legible clean slip + fee paid + valid sponsor +
registry CLEAR) is only 65% APPROVED, and 8*0.65 - 4*0.23 still beats routing to
review.  If a reviewer prefers fewer catastrophic approvals over raw score, set
``fa_penalty`` harder; the TRAIN OOF sweep prices the trade exactly (with an
isotonic-calibrated model, -20 buys 10 false approvals instead of 36 for -1.7
points).  The default is whatever maximised TRAIN OOF.

Compliance notes
----------------
*   Nothing is keyed to a case_id.  The only literal identifiers in this file are
    the three revoked sponsor IDs printed in the public FIELD_MANUAL.
*   The model, the isotonic calibrator and every threshold are fit on TRAIN_IDS
    only; DEV_IDS is touched exactly once, at report time.
*   Only ``common.doc_lines`` (visible OCR) is read.  ``hidden_text`` is never a
    value source; it reaches the model only through extract_v2's
    *disagreement counts*, which are a damage/tamper measurement.
*   Graceful degradation: every note/policy signal is an *additive* feature.  A
    private packet with an unseen layout simply has them all zero and falls back
    to the document-shape + resolution-quality features.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common  # noqa: E402

CLASSES = ("APPROVED", "DENIED", "NEEDS_REVIEW")
MODEL_PATH = _HERE / "clf_d_model.joblib"

# ---------------------------------------------------------------------------
# 1. adjudicator-note reader
# ---------------------------------------------------------------------------

# thresholds below were chosen on TRAIN_IDS only
_FIND_ANCHOR_THR = 0.72
_FIND_VALUE_THR = 0.62
_REASON_THR = 0.78

_DECISION_WORDS = {"denied": "DENIED", "approved": "APPROVED",
                   "needsreview": "NEEDS_REVIEW"}

# Closed set of Reason templates emitted by the MIB adjudication system.  These
# are policy sentences, not case data; an unseen one just matches nothing.
_REASON_TEMPLATES = [
    ("dq_flag",       "disqualifyingriskflag"),
    ("ro_flag",       "reviewonlyriskflagpresent"),
    ("transit",       "transitclasscannotauthorize"),
    ("embargo_world", "embargohomeworld"),
    ("revoked_spn",   "revokedsponsor"),
    ("fee_unpaid",    "mandatoryfeeunpaid"),
    ("fee_unknown",   "feestatusunknown"),
    ("date_missing",  "arrivaldatemissingfromtrustedvisibleevidence"),
    ("damaged",       "packetcontainsdamagedorcontradictoryvisibleevidence"),
    ("clean",         "cleanorexceptionqualifiedpacket"),
    ("appr_survive",  "approvalsupportedbysurvivingvisibleevidence"),
    ("deny_damaged",  "denialsupportedbydamagedregistryevidence"),
    ("ambiguous",     "ambiguouspacket"),
    ("rescinded",     "priordenialstamprescinded"),
]

# Sponsors the *public field manual* lists as revoked.  Policy, not mined data.
MANUAL_REVOKED_SPONSORS = ("SPN-0007", "SPN-0139", "SPN-4040")


def _sq(s: str) -> str:
    """lowercase alphanumerics only - OCR both invents and eats whitespace."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _best_window(pat: str, s: str, lo: float = 0.6, hi: float = 1.6):
    """(ratio, start, end) of the substring of `s` most similar to `pat`."""
    n = len(pat)
    if not s or not pat:
        return 0.0, 0, 0
    best = (0.0, 0, 0)
    sm = difflib.SequenceMatcher()
    sm.set_seq2(pat)
    for w in range(max(3, int(n * lo)), int(n * hi) + 1):
        if w > len(s):
            if best[0] == 0.0:
                sm.set_seq1(s)
                best = (sm.ratio(), 0, len(s))
            break
        for i in range(len(s) - w + 1):
            sm.set_seq1(s[i:i + w])
            if sm.real_quick_ratio() <= best[0] or sm.quick_ratio() <= best[0]:
                continue
            r = sm.ratio()
            if r > best[0]:
                best = (r, i, i + w)
    return best


def note_signals(rec) -> dict:
    """Fuzzy-read the visible Manual Adjudicator Note. Visible OCR only.

    Returns {'finding': str|None, 'finding_score': float, 'reasons': {tag: score}}.
    """
    lines = common.doc_lines(rec)
    sqs = [_sq(l["t"]) for l in lines]

    best_find = (0.0, None)
    for i, s in enumerate(sqs):
        if len(s) < 5:
            continue
        r, _a, b = _best_window("finding", s)
        if r < _FIND_ANCHOR_THR:
            continue
        tail = s[b:b + 18]
        if len(tail) < 4 and i + 1 < len(sqs):          # value wrapped to next line
            tail = sqs[i + 1][:18]
        for word, dec in _DECISION_WORDS.items():
            rr, _, _ = _best_window(word, tail, 0.7, 1.3)
            if rr >= _FIND_VALUE_THR and min(r, rr) > best_find[0]:
                best_find = (min(r, rr), dec)

    # Reason templates are always introduced by a literal "Reason:" on the note,
    # so anchor on that first and only fuzzy-compare the text that follows.  This
    # keeps the reader at ~14 string comparisons per note instead of a full
    # sliding-window search over every line of the packet.
    tails = []
    for i, s in enumerate(sqs):
        if len(s) < 6:
            continue
        r, _a, b = _best_window("reason", s, 0.8, 1.4)
        if r < 0.75:
            continue
        tail = s[b:]
        if len(tail) < 8:                               # reason wrapped to next line
            tail = (tail + sqs[i + 1]) if i + 1 < len(sqs) else tail
        elif i + 1 < len(sqs):
            tails.append(tail + sqs[i + 1])             # allow a wrapped suffix too
        tails.append(tail)

    reasons = {}
    sm = difflib.SequenceMatcher()
    for tag, tpl in _REASON_TEMPLATES:
        sm.set_seq2(tpl)
        best = 0.0
        for tail in tails:
            sm.set_seq1(tail[:int(len(tpl) * 1.25)])
            if sm.real_quick_ratio() <= best:
                continue
            r = sm.ratio()
            if r > best:
                best = r
        if best >= _REASON_THR:
            reasons[tag] = best

    return {"finding": best_find[1], "finding_score": best_find[0],
            "reasons": reasons}


# ---------------------------------------------------------------------------
# 2. policy features
# ---------------------------------------------------------------------------

def slip_features(rec) -> dict:
    """Biometric-slip evidence *quality*.

    flags.py answers 'none' both when the slip legibly enumerated no flags and
    when the packet carries no risk evidence at all.  Those two states have very
    different decision priors, so expose them separately.
    """
    import flags as flags_mod
    try:
        _f, ev = flags_mod.detect_flags_detail(rec)
    except Exception:
        return {"slip_anchor": 0, "slip_resolved": 0, "slip_n_flags": 0,
                "slip_clean": 0, "note_flag_n": 0, "slip_name_conflict": 0,
                "evidence_none": 1}
    return {
        "slip_anchor": int(bool(ev["slip_anchor"])),
        "slip_resolved": int(bool(ev["slip_resolved"])),
        "slip_n_flags": len(ev["slip_flags"]),
        # the packet positively asserts a clean risk posture
        "slip_clean": int(bool(ev["slip_anchor"]) and bool(ev["slip_resolved"])
                          and not ev["slip_flags"]),
        "note_flag_n": len(ev["note_flags"]),
        "slip_name_conflict": int(bool(ev["name_conflict"])),
        # no risk evidence anywhere -> 'none' is a guess, not an observation
        "evidence_none": int(not ev["anchor"]),
    }


# Literal form strings emitted by the MIB intake forms.  Closed policy vocabulary
# (fee statuses, visa classes, registry statuses), never applicant data.
_RAW_STRINGS = {
    # NB: patterns are matched against the *squashed* (alphanumeric-only) visible
    # text, so they must themselves contain no punctuation.
    "raw_fee_unpaid": ("feestatusunpaid",),
    "raw_fee_unknown": ("feestatusunknown",),
    "raw_fee_paid": ("feestatuspaid",),
    "raw_fee_waived": ("feestatuswaived",),
    "raw_waivercode_na": ("waivercodena",),
    "raw_visa_transit": ("visaclasstransit", "classtransit"),
    "raw_visa_dip": ("visaclassdip", "classdip1"),
    "raw_observed_any": ("observedflags",),
    "raw_observed_none": ("observedflagsnone",),
    "raw_registry_clear": ("registrystatusclear",),
    "raw_registry_embargo": ("registrystatusembargoreview", "embargoreview"),
    "raw_diplomatic": ("diplomatic",),
    "raw_hardship": ("hardship",),
    "raw_rescind": ("rescind",),
    "raw_revoked": ("revoked",),
    "raw_sponsor_standing": ("sponsorstandingrequiresadditionalverification",
                             "standingrequiresadditional"),
    "raw_amount_zero": ("amount000", "amountna"),
}


def raw_form_features(visible_sq: str) -> dict:
    """Direct closed-vocabulary form evidence, independent of the extractor.

    fee_status/visa_class resolution is only ~82%/90% accurate, and those two
    fields drive three FIELD_MANUAL rules, so give the model the raw form string
    as well as the extractor's opinion and let it arbitrate.
    """
    return {k: int(any(p in visible_sq for p in pats))
            for k, pats in _RAW_STRINGS.items()}


_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def date_features(rec, fields) -> dict:
    """FIELD_MANUAL staleness rule, expressed *relative to the packet itself*.

    "Applications are stale if the arrival date is more than 180 days before
    packet receipt."  An absolute calendar feature would not survive a private
    set generated in a different date window, so measure the arrival date against
    the other dates printed on the same packet instead.  Shift-invariant.
    """
    from datetime import date
    ds = []
    for m in _DATE_RE.finditer(common.visible_text(rec)):
        try:
            ds.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    out = {"date_n_distinct": len(set(ds)), "date_span_days": 0.0,
           "date_arrival_lag_days": 0.0, "date_stale_180": 0}
    if not ds:
        return out
    out["date_span_days"] = float((max(ds) - min(ds)).days)
    m = _DATE_RE.fullmatch(str(fields.get("arrival_date", "")))
    if m:
        try:
            arr = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return out
        lag = (max(ds) - arr).days
        out["date_arrival_lag_days"] = float(lag)
        out["date_stale_180"] = int(lag > 180)
    return out


def policy_features(rec, fields) -> dict:
    """FIELD_MANUAL rules the stock 97-vector cannot express."""
    visible = _sq(common.visible_text(rec))
    sponsor = str(fields.get("sponsor_id", ""))
    visa = str(fields.get("visa_class", ""))
    fee = str(fields.get("fee_status", ""))
    flags = set(str(fields.get("risk_flags", "none")).split("|")) - {"none", ""}

    manual_revoked = int(sponsor in MANUAL_REVOKED_SPONSORS)
    # visible prose asserting the sponsor is not in good standing
    revoked_prose = int(
        "revokedsponsor" in visible
        or "sponsorstandingrequiresadditionalverification" in visible
        or "sponsorrevoked" in visible)
    embargo_registry = int("registrystatusembargoreview" in visible
                           or "embargoreview" in visible)

    return {
        "pol_manual_revoked_sponsor": manual_revoked,
        "pol_revoked_prose": revoked_prose,
        "pol_revoked_any": int(manual_revoked or revoked_prose),
        "pol_registry_embargo": embargo_registry,
        "pol_registry_clear": int("registrystatusclear" in visible),
        "pol_transit": int(visa == "TRANSIT-7"),
        "pol_dip": int(visa == "DIP-1"),
        "pol_med3": int(visa == "MED-3"),
        # fee policy: waived is only acceptable for DIP-1 or a visible hardship waiver
        "pol_waived_ok": int(fee == "waived" and (visa == "DIP-1"
                                                  or "hardship" in visible)),
        "pol_waived_bad": int(fee == "waived" and visa != "DIP-1"
                              and "hardship" not in visible),
        "pol_unpaid_no_waiver": int(fee == "unpaid" and "waiver" not in visible),
        "pol_fee_unknown": int(fee == "unknown"),
        # sponsor required unless DIP-1
        "pol_sponsor_missing": int(
            visa != "DIP-1" and not re.fullmatch(r"SPN-\d{4}", sponsor)),
        "pol_date_missing": int(str(fields.get("arrival_date")) in
                                ("", "1900-01-01", "unknown", "None")),
        "pol_dq_flag": int(bool(flags & common.DISQUALIFYING)),
        "pol_ro_flag_n": len(flags & common.REVIEW_ONLY),
        "pol_ro_multi": int(len(flags & common.REVIEW_ONLY) >= 2),
        # documented traps, so the model can learn to *discount* them
        "pol_sample_denial_wm": int("sampledenial" in visible or
                                    "sampledenia" in visible),
        "pol_hardship_visible": int("hardship" in visible),
        "pol_diplomatic_note": int("diplomaticnote" in visible),
        "pol_med3_no_biohazard_evidence": int(
            visa == "MED-3" and "observedflags" not in visible),
    }


# ---------------------------------------------------------------------------
# 3. full feature vector
# ---------------------------------------------------------------------------

def design_features(rec, fields, base_feats) -> dict:
    """base_feats = extract_v2.extract_features(rec, fields) (the 97 stock keys)."""
    ns = note_signals(rec)
    f = dict(base_feats)
    f.update(policy_features(rec, fields))
    f.update(slip_features(rec))
    f.update(raw_form_features(_sq(common.visible_text(rec))))
    f.update(date_features(rec, fields))
    for c in CLASSES:
        f[f"note_find_{c}"] = int(ns["finding"] == c)
    f["note_find_any"] = int(ns["finding"] is not None)
    f["note_find_score"] = float(ns["finding_score"])
    for tag, _ in _REASON_TEMPLATES:
        f[f"note_reason_{tag}"] = float(ns["reasons"].get(tag, 0.0))
    f["note_reason_any"] = int(bool(ns["reasons"]))
    f["note_n_reasons"] = len(ns["reasons"])
    # does the note contradict the flag detector? (damaged / adversarial packet)
    f["note_find_vs_dqflag"] = int(
        (ns["finding"] == "APPROVED") and f.get("pol_dq_flag", 0) == 1)
    return f


# ---------------------------------------------------------------------------
# 4. expected-value decision rule (EVALUATION.md scoring table)
# ---------------------------------------------------------------------------
# rows = action, cols = truth (APPROVED, DENIED, NEEDS_REVIEW)
def payoff(fa_penalty: float = -4.0) -> np.ndarray:
    return np.array([
        [8.0, fa_penalty, 1.0],   # predict APPROVED : correct / FALSE APPROVAL / missed NR
        [0.0,        8.0, 1.0],   # predict DENIED   : wrong A-vs-D / correct / missed NR
        [2.0,        2.0, 8.0],   # predict NEEDS_REVIEW
    ])


PAYOFF = payoff()

# The literal table says -4.  A *harsher* internal penalty buys a lower
# catastrophic-false-approval count (EVALUATION.md tie-breaker #2 and the
# "no catastrophic false-approval pattern" bar) for ~zero expected points,
# because the marginal approvals it gives up are near-coin-flips.  The value is
# chosen on TRAIN out-of-fold, never on dev.
FA_PENALTY_GRID = (-4.0, -6.0, -8.0, -12.0, -20.0)


def decide(proba: np.ndarray, fa_penalty: float = -4.0):
    """proba (n,3) -> (labels, confidence). Maximises expected challenge points."""
    ev = proba @ payoff(fa_penalty).T           # (n, 3) expected points per action
    act = ev.argmax(1)
    conf = proba[np.arange(len(proba)), act]    # P(chosen action is the truth)
    return [CLASSES[a] for a in act], np.clip(conf, 0.01, 0.99)


# ---------------------------------------------------------------------------
# 5. model
# ---------------------------------------------------------------------------

def challenge_score(truth, pred, conf):
    """Replicate EVALUATION.md's classification + calibration terms exactly.

    Verified against scripts/evaluate.py (see __main__ output): identical to 2dp.
    Used for TRAIN-only model selection so DEV is never fit on.
    """
    raw = 0
    fa = 0
    for t, p in zip(truth, pred):
        if t == p:
            raw += 8
        elif p == "NEEDS_REVIEW":
            raw += 2
        elif t == "NEEDS_REVIEW":
            raw += 1
        elif t == "DENIED" and p == "APPROVED":
            raw -= 4
            fa += 1
    cls = 80.0 * raw / (8.0 * len(truth))
    brier = float(np.mean([(c - (t == p)) ** 2
                           for t, p, c in zip(truth, pred, conf)]))
    return {"classification": cls, "calibration": 20.0 * max(0.0, 1 - 2 * brier),
            "false_approvals": fa, "accuracy": np.mean(
                [t == p for t, p in zip(truth, pred)])}


# candidate configurations; the winner is picked by TRAIN out-of-fold score only
_CONFIGS = [
    {"calib": "isotonic", "max_iter": 400, "lr": 0.06, "leaves": 31},
    {"calib": "sigmoid", "max_iter": 400, "lr": 0.06, "leaves": 31},
    {"calib": None, "max_iter": 400, "lr": 0.06, "leaves": 31},
    {"calib": None, "max_iter": 250, "lr": 0.05, "leaves": 15},
    {"calib": "sigmoid", "max_iter": 250, "lr": 0.05, "leaves": 15},
]


class Adjudicator:
    """Fit on TRAIN_IDS only. Predict = calibrated probabilities + EV decision."""

    def __init__(self, seed: int = 0, config: dict | None = None,
                 fa_penalty: float = -4.0, rule: str = "ev"):
        self.seed = seed
        self.config = dict(config or _CONFIGS[0])
        self.fa_penalty = fa_penalty
        self.rule = rule
        self.keys: list[str] = []
        self.model = None

    def _make(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        c = self.config
        base = HistGradientBoostingClassifier(
            max_iter=c["max_iter"], learning_rate=c["lr"],
            max_leaf_nodes=c["leaves"], min_samples_leaf=20,
            l2_regularization=1.0, random_state=self.seed)
        if c["calib"] is None:
            return base
        return CalibratedClassifierCV(base, method=c["calib"], cv=5)

    # -- plumbing ----------------------------------------------------------
    def _matrix(self, feats_list):
        return np.array([[float(d.get(k, 0.0)) for k in self.keys]
                         for d in feats_list], dtype=np.float64)

    def fit(self, feats_list, labels, keys=None):
        self.keys = keys or sorted({k for d in feats_list for k in d})
        X = self._matrix(feats_list)
        y = np.array([CLASSES.index(l) for l in labels])
        self.model = self._make()
        self.model.fit(X, y)
        return self

    @staticmethod
    def oof_proba(feats_list, labels, config, seed=0, folds=5):
        """Out-of-fold probabilities on TRAIN, for honest model selection."""
        from sklearn.model_selection import StratifiedKFold
        keys = sorted({k for d in feats_list for k in d})
        y = np.array([CLASSES.index(l) for l in labels])
        out = np.zeros((len(feats_list), len(CLASSES)))
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for a, b in skf.split(np.zeros(len(y)), y):
            m = Adjudicator(seed=seed, config=config)
            m.fit([feats_list[i] for i in a], [labels[i] for i in a], keys=keys)
            out[b] = m.predict_proba([feats_list[i] for i in b])
        return out

    def predict_proba(self, feats_list):
        return self.model.predict_proba(self._matrix(feats_list))

    def predict(self, feats_list):
        p = self.predict_proba(feats_list)
        if self.rule == "argmax":
            return ([CLASSES[i] for i in p.argmax(1)],
                    np.clip(p.max(1), 0.01, 0.99))
        return decide(p, self.fa_penalty)

    def predict_one(self, rec, fields, base_feats):
        """Convenience end-to-end entry point for the production pipeline."""
        f = design_features(rec, fields, base_feats)
        lab, conf = self.predict([f])
        return lab[0], float(conf[0])

    # -- persistence -------------------------------------------------------
    def save(self, path=MODEL_PATH):
        import joblib
        joblib.dump({"keys": self.keys, "model": self.model,
                     "fa_penalty": self.fa_penalty, "rule": self.rule,
                     "config": self.config}, path, compress=3)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        import joblib
        d = joblib.load(path)
        o = cls(config=d.get("config"), fa_penalty=d.get("fa_penalty", -4.0),
                rule=d.get("rule", "ev"))
        o.keys, o.model = d["keys"], d["model"]
        return o


# ---------------------------------------------------------------------------
# 6. self-test / report
# ---------------------------------------------------------------------------

def _load_featcache():
    """fields + stock 97 features for every cached case (built once, reused)."""
    import pickle
    pkl = _HERE.parent / "featcache_d.pkl"
    if pkl.exists():
        return pickle.load(open(pkl, "rb"))
    import extract_v2
    cache = common.load_cache()
    out = {}
    for cid, rec in sorted(cache.items()):
        fl = extract_v2.extract_fields_v2(rec)
        out[cid] = (fl, extract_v2.extract_features(rec, fl))
    pickle.dump(out, open(pkl, "wb"))
    return out


def _design_cache(cache, fc, ids):
    import pickle
    pkl = _HERE.parent / "designfeat_d.pkl"
    have = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    todo = [c for c in ids if c not in have]
    for n, c in enumerate(todo):
        have[c] = design_features(cache[c], fc[c][0], fc[c][1])
        if n and n % 200 == 0:
            print(f"    ... design features {n}/{len(todo)}", flush=True)
    if todo:
        pickle.dump(have, open(pkl, "wb"))
    return have


def _main():
    labels = common.load_labels()
    cache = common.load_cache()
    fc = _load_featcache()
    train, dev = common.split_ids(labels)
    train = [c for c in train if c in cache and c in fc]
    dev = [c for c in dev if c in cache and c in fc]
    print(f"cache={len(cache)}  train={len(train)}  dev={len(dev)}")

    print("  building design features ...", flush=True)
    D = _design_cache(cache, fc, train + dev)

    # ---- note reader precision on TRAIN (evidence-quality sanity check)
    hit = ok = 0
    for c in train:
        d = D[c]
        f = next((k for k in CLASSES if d[f"note_find_{k}"]), None)
        if f:
            hit += 1
            ok += (f == labels[c]["adjudication"])
    print(f"  note 'Finding:' reader on TRAIN: fires {hit}/{len(train)} "
          f"({100*hit/len(train):.0f}%), correct {ok}/{hit} "
          f"({100*ok/max(hit,1):.1f}%)")

    # ---- model selection: TRAIN out-of-fold only, DEV untouched
    Ftr = [D[c] for c in train]
    Ytr = [labels[c]["adjudication"] for c in train]
    print("  TRAIN out-of-fold model selection (dev never touched):")
    best = None
    for cfg in _CONFIGS:
        pr = Adjudicator.oof_proba(Ftr, Ytr, cfg)
        am = [CLASSES[i] for i in pr.argmax(1)]
        s_am = challenge_score(Ytr, am, pr.max(1))
        line = (f"    {str(cfg['calib']):9s} it={cfg['max_iter']:3d} "
                f"lv={cfg['leaves']:2d} | argmax "
                f"{s_am['classification']+s_am['calibration']:6.2f}"
                f" (fa {s_am['false_approvals']:3d}) |")
        if s_am["classification"] + s_am["calibration"] > (best[0] if best else -1):
            best = (s_am["classification"] + s_am["calibration"], cfg, -4.0, "argmax")
        for pen in FA_PENALTY_GRID:
            lab, cf = decide(pr, pen)
            s = challenge_score(Ytr, lab, cf)
            tot = s["classification"] + s["calibration"]
            line += f" EV{int(-pen):02d} {tot:6.2f}(fa {s['false_approvals']:3d})"
            if tot > best[0]:
                best = (tot, cfg, pen, "ev")
        print(line)
    _tot, cfg, pen, rule = best
    print(f"  -> selected {cfg}, rule={rule}, fa_penalty={pen} "
          f"(TRAIN OOF cls+cal {_tot:.2f})")

    clf = Adjudicator(config=cfg, fa_penalty=pen, rule=rule).fit(Ftr, Ytr)
    print(f"  fitted on {len(train)} TRAIN cases, {len(clf.keys)} features")

    # ---- DEV report (measured once, never fit on)
    proba = clf.predict_proba([D[c] for c in dev])
    argmax = [CLASSES[i] for i in proba.argmax(1)]
    if rule == "argmax":
        adj, conf = argmax, np.clip(proba.max(1), 0.01, 0.99)
    else:
        adj, conf = decide(proba, pen)

    truth = [labels[c]["adjudication"] for c in dev]
    print(f"  dev accuracy: argmax {np.mean(np.array(argmax)==np.array(truth)):.3f}"
          f"   EV-rule {np.mean(np.array(adj)==np.array(truth)):.3f}")
    conf_mat = {}
    for t, p in zip(truth, adj):
        conf_mat[(t, p)] = conf_mat.get((t, p), 0) + 1
    print("  dev confusion (truth -> pred):")
    for t in CLASSES:
        print("   ", t.ljust(13),
              "  ".join(f"{p[:4]}={conf_mat.get((t,p),0):3d}" for p in CLASSES))

    rows = []
    for c, a, cf in zip(dev, adj, conf):
        r = {"case_id": c, **{k: v for k, v in fc[c][0].items() if k != "_meta"}}
        r["adjudication"] = a
        r["confidence"] = round(float(cf), 4)
        rows.append(r)
    res = common.score_rows(rows, "clf_d_dev")
    print("  DEV  " + common.summarize(res))

    # ---- reference: same rows scored with argmax + argmax-prob confidence
    rows2 = []
    for c, a, pr in zip(dev, argmax, proba):
        r = {"case_id": c, **{k: v for k, v in fc[c][0].items() if k != "_meta"}}
        r["adjudication"] = a
        r["confidence"] = round(float(pr.max()), 4)
        rows2.append(r)
    res2 = common.score_rows(rows2, "clf_d_dev_argmax")
    print("  (ref argmax) " + common.summarize(res2))

    clf.save()
    print(f"  saved model -> {MODEL_PATH} "
          f"({MODEL_PATH.stat().st_size/1e6:.1f} MB)")
    return res


if __name__ == "__main__":
    _main()
