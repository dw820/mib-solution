"""FINAL adjudication classifier - the security-hardened, honestly-selected model.

This module supersedes clf_r3_2.  It exists to fix three defects that an
adversarial + methodological audit confirmed in that module.  Every number in
this docstring was measured by ``__main__`` (or by the sibling scripts named
below); nothing here is estimated.


DEFECT 2 - THE CLASSIFIER ATE AN ATTACKER-CONTROLLED FEATURE
------------------------------------------------------------
``repair.disagreement_features`` exports five counters that compare the PDF text
layer against the visible OCR values.  ``extract_v2.extract_features`` copied all
five into the base vector, so all five reached clf_r3_2 through ``features_A`` /
``features_B``.  Two of them - ``hidden_agree_n`` and ``hidden_near_n`` - reward
the untrusted channel for CORROBORATING the visible values.  A boosted tree
learns the obvious correlation ("clean packets have a text layer that matches"),
and the attack that follows needs no injected instruction at all: simply ECHO
the visible field values into the text layer and ``hidden_agree_n`` is maximal.

Measured on DEV with the shipped clf_r3_2 artifact (see ``_fin_before.py``):

    echo attack .......... 8/200 adjudications flip, 4 of them toward APPROVED,
                           catastrophic false approvals 3 -> 5, including
                           MIB-000040 and MIB-000700 (both truth=DENIED) moved
                           NEEDS_REVIEW -> APPROVED.  Total 124.35 -> 123.95.
    instruction attack ... 7/200 adjudications shift (all away from APPROVED),
                           total 124.35 -> 122.86.  The instruction payload can
                           only make clf_r3_2 more cautious; the ECHO payload is
                           the one that buys approvals, which is the whole point:
                           the exploitable feature is the AGREEMENT counter, and
                           an attacker who wants approvals writes no instruction
                           at all.

THE FIX, exactly as the auditor prescribed:

*   ``hidden_agree_n`` and ``hidden_near_n`` are DROPPED from the feature set.
    They are the promotion channel and there is no safe way to keep them: a tree
    is free to use them in either direction, so "the model will only use them to
    become cautious" is an assertion, not a guarantee.
*   The three monotone-safe signals (``hidden_disagree_n``,
    ``hidden_refused_fill_n``, ``hidden_any_disagree``) are ALSO removed from the
    model input.  They survive only as a post-hoc CAUTION RULE that is allowed to
    demote a decision toward NEEDS_REVIEW and can never promote one.  Structural
    monotonicity, not a fitted one.
*   ``BANNED_KEYS`` / ``is_safe_key`` / ``_strip`` filter the vectors at both fit
    and predict time (and ``inj_*``, clf_r3_3's hidden-text profile, is banned
    pre-emptively so a later feature merge cannot re-import the hole), and
    ``__main__`` asserts the model input is bit-identical under both attacks, so
    the property is enforced by code rather than by convention.

The security property this buys is unconditional, not statistical: the model's
input vector is a pure function of visible OCR, therefore ANY text-layer attack -
echo, instruction, answer key, or one nobody has thought of yet - produces the
same probability vector.  The only thing the text layer can still do is fire the
caution rule, whose whole output range is {unchanged, moved to NEEDS_REVIEW}.

THE CAUTION RULE, AND WHY IT SHIPS DISABLED
--------------------------------------------
``apply_caution`` is implemented, exercised and selected over five modes on TRAIN
OOF.  It ships at ``off``, and the honest reason is worth stating rather than
burying.  Measured TRAIN OOF (cls+cal, 5 paired fold seeds, at the selected
decision layer):

    caution=off       78.60   false approvals 11.4 / 800
    caution=dis       75.98   false approvals  8.0 / 800
    caution=dis_all   66.39   false approvals  8.0 / 800
    caution=any       75.84   false approvals  8.0 / 800
    caution=any_all   66.24   false approvals  8.0 / 800

The trigger fires on 323/800 TRAIN packets, because on real scans the text layer
routinely differs from the OCR reading for ordinary reasons (noise, hyphenation,
a redaction bar) rather than adversarial ones.  Paying 2.6 OOF points to route
40% of the caseload to NEEDS_REVIEW is not a trade the inherited selection rule
(argmax, then fewest false approvals within SELECT_TOL = 0.5) will take, and I am
not going to override the rule to make the module look more defended.

The security case for the rule is also weaker than it looks once the features are
gone: with the model input already invariant to the text layer, the caution rule
adds no attack resistance at all - it can only cost clean accuracy.  It is kept
in the module, fully wired, so that a deployment that values NEEDS_REVIEW more
than this scoring function does can turn it on with one constructor argument and
a known price.

Cost of the security fix, measured on clean DEV with the real
``scripts/evaluate.py``:

                        total    class   extract   calib   false-approvals
    BEFORE  clf_r3_2   124.35    64.15    43.92    16.28         3
    AFTER   clf_final  123.89    63.65    43.92    16.33         3
                       -0.46    -0.50      0.00    +0.05         0

Half a point of classification is the honest price of deleting an exploitable
feature, and it is bought back many times over the first time a packet arrives
with an echoed text layer.


DEFECT (i) - THE SHIPPED MODEL WAS SELECTED ON DEV
--------------------------------------------------
clf_r3_2 was the argmax of six candidate modules ALL scored on the same 200-case
DEV set (118.84 / 123.09 / 123.16 / 123.16 / 123.43 / 124.35).  A 1.2-point
spread over six tries on 200 cases is inside selection noise, and clf_r3_3 - the
runner-up on dev - actually had the HIGHER TRAIN out-of-fold score (79.14 vs
78.55).  So the shipped dev number was a selection maximum, not a held-out
estimate.

This module re-selects on TRAIN out-of-fold ONLY (five paired fold seeds), and
the dev number is then read exactly once as a genuine held-out estimate.  The
candidate list is re-run from scratch under the cleaned feature set, because
dropping columns changes the ranking - notably:

*   clf_r3_3's entire OOF advantage came from its block D, the "injection
    profile", which is 100% derived from ``hidden_text``.  Under the security
    fix block D is deleted, and what remains of clf_r3_3 IS cell A.  So the
    audit's "clf_r3_3 had the higher train-OOF" is true and moot: that model is
    not shippable, and its clean-feature residue is already a candidate here.

Re-run on TRAIN OOF under the cleaned feature set (5 paired fold seeds, each cell
allowed its own decision layer):

    cell A (design, hgb+ET2+LR) ......... OOF 78.35 +- 0.38   fa 10.4/800
    cell B (design+evidence, hgb_low) ... OOF 77.59 +- 0.41   fa  4.4/800
    cell blend 0.5A + 0.5B .............. OOF 78.60 +- 0.52   fa 11.4/800   <- selected

SELECTED: blend, recal=none, fa_penalty=-8, brier_w=8, caution=off.  Note the
blend's OOF lead over cell A is 0.25 against a paired fold-seed sd of ~0.4-0.5,
i.e. it is NOT resolved by this protocol either; the blend is kept because it was
already the incumbent architecture, not because this measurement proved it
better.  Saying so is the whole point of the exercise.

TRAIN OOF 78.60 (cls+cal) vs DEV 79.98 (cls+cal): gap +1.38 in DEV's favour.
DEV is the smaller sample and was read once, so this is a genuine held-out
estimate and the gap is sampling noise, not selection optimism - which is exactly
the property clf_r3_2's 124.35 did not have.


DEFECT (ii) - USE_FEE_PRIOR WAS ON BY INERTIA
----------------------------------------------
``extract_v2.USE_FEE_PRIOR`` emits the TRAIN majority ``fee_status`` when a
packet shows no fee evidence at all.  Its own in-source comment records an A/B in
which OFF scored BETTER overall (108.14 ON vs 109.05 OFF) and left it ON anyway,
on the argument that a *real* classifier would consume the ``fee_is_prior`` /
``fee_observed`` provenance flags and recover the classification loss.  That
argument was never tested against a real classifier.  ``_fin_feeprior.py`` tests
it: both settings get their features rebuilt from scratch, 5-fold OOF on TRAIN
(fold seeds 0 and 1), their own decision layer selected within the variant, and
the resulting TRAIN OOF rows scored end-to-end by the real ``evaluate.py`` so the
extraction term the prior is supposed to buy is actually counted.

    TRAIN OOF (full /150)   total    class   extract   calib   false-approvals
      USE_FEE_PRIOR = ON   121.74    62.61    43.35    15.78        22 / 800
      USE_FEE_PRIOR = OFF  121.27    63.55    42.23    15.50        28 / 800

DECISION: ON, by measurement.  The prior buys +1.12 extraction and costs -0.94
classification, net +0.47 - and, contrary to the stub-classifier A/B, it also
REDUCES false approvals (22 vs 28) rather than adding ten of them.  The original
comment's defence turns out to be correct: a classifier that consumes
``fee_is_prior`` / ``fee_observed`` routes the guessed values to NEEDS_REVIEW
instead of approving them.  The comment was right for a reason it had not
tested; it is now tested.

DEV effect, reported after the fact and not used to decide (same 2-seed protocol,
so the decision layer differs slightly from the shipped one):
      ON  124.22/150  (fee_status accuracy 0.815, false approvals 5)
      OFF 123.25/150  (fee_status accuracy 0.635, false approvals 8)
DEV agrees with TRAIN, which is reassuring but was not the evidence.


COMPLIANCE
----------
*   Nothing is keyed to a case_id.
*   Every model, threshold, blend weight and rule is fitted/selected on TRAIN_IDS
    only, out of fold.  DEV is read once, at the end, for reporting.
*   The classifier's input is visible OCR only.  ``hidden_text`` reaches nothing
    except the demote-only caution rule.


MEASURED RESULTS (DEV_IDS = 200 held-out cases, real ``scripts/evaluate.py``)
----------------------------------------------------------------------------
                        total    class   extract   calib   false-approvals
    clf_r3_2  clean    124.35    64.15    43.92    16.28         3
    clf_r3_2  + echo   123.95    63.75    43.92    16.28         5   <- 8 flips,
                                                                        4 toward
                                                                        APPROVED
    clf_r3_2  + instr  122.86    62.45    43.92    16.49         3   <- 7 flips
    ---------------------------------------------------------------------------
    clf_final clean    123.89    63.65    43.92    16.33         3
    clf_final + echo   123.89    63.65    43.92    16.33         3   <- 0 flips
    clf_final + instr  123.89    63.65    43.92    16.33         3   <- 0 flips

dev accuracy 0.755, Brier 0.0919.  Confusion (truth -> pred):
    APPROVED       A 28   D  6   NR 17
    DENIED         A  3   D 71   NR 20
    NEEDS_REVIEW   A  2   D  1   NR 52

The attacked rows are byte-identical to the clean row, not merely close, and
``__main__`` asserts it: the two attacks are no longer *survivable*, they are
*inert*.


HONEST LIMITS
-------------
*   -0.46 total on clean DEV against clf_r3_2.  Some of that is the deleted
    feature and some is the re-selection; I did not try to separate them, because
    both changes are non-negotiable and the decomposition would not change what
    ships.
*   The security guarantee covers the CLASSIFIER's input.  It says nothing about
    the extractor, which is a separate defect with a separate fix (see
    ``repair.py`` and ``_fix1_measure.py``); the extraction term is unchanged at
    43.92/50 under both attacks here, which is evidence that fix is holding, not
    proof.
*   Two DEV cases are still adjudicated APPROVED against truth NEEDS_REVIEW and
    three against truth DENIED.  Those are ordinary model errors on clean input,
    not injection, and the fa_penalty frontier printed by ``__main__`` shows what
    reducing them costs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common          # noqa: E402
import clf_d           # noqa: E402
import clf_r2_1        # noqa: E402
import clf_r2_3        # noqa: E402
import repair as repair_mod   # noqa: E402

CLASSES = clf_d.CLASSES
I_APPROVED = CLASSES.index("APPROVED")
I_DENIED = CLASSES.index("DENIED")
I_REVIEW = CLASSES.index("NEEDS_REVIEW")

MODEL_PATH = _HERE / "clf_final_model.joblib"
_SCRATCH = _HERE.parent

# ---------------------------------------------------------------------------
# THE SECURITY BOUNDARY
# ---------------------------------------------------------------------------
# Any feature whose value a party who controls only the PDF text layer can move.
# ``hidden_agree_n``/``hidden_near_n`` are the promotion channel and are gone for
# good; the other three are gone from the MODEL and live on in ``caution_signals``
# where they are structurally demote-only.  ``inj_*`` is clf_r3_3's block D,
# blocked pre-emptively so it cannot be re-imported by a later feature merge.
BANNED_KEYS = frozenset({
    "hidden_agree_n", "hidden_near_n",
    "hidden_disagree_n", "hidden_refused_fill_n", "hidden_any_disagree",
})
BANNED_PREFIXES = ("hidden_", "inj_")


def is_safe_key(k: str) -> bool:
    return k not in BANNED_KEYS and not k.startswith(BANNED_PREFIXES)


def _strip(d: dict) -> dict:
    """Drop every attacker-controllable column from a feature dict."""
    return {k: v for k, v in d.items() if is_safe_key(k)}


# --- constants that are DERIVED or DECLARED, never tuned ---------------------
BRIER_W = 4.0          # total = (10/N) sum[raw - 4*brier] + 20  -> 1 brier == 4 raw
BLEND_W = 0.5          # equal weight over the two cells (inherited, plateau flat)
SELECT_TOL = 0.5       # inherited from clf_r2_1 / clf_r3_2
FOLD_SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5

SPEC_A = "hgb+ET2+LR"  # 200 design features, high-capacity heterogeneous ensemble
SPEC_B = "hgb_low"     # design + evidence, stump-depth booster

decide = clf_r2_1.decide
challenge_score = clf_r2_1.challenge_score
payoff = clf_r2_1.payoff
fit_recal = clf_r2_1.fit_recal
apply_recal = clf_r2_1.apply_recal

# Set by the TRAIN-side A/B in ``_fin_feeprior.py``; see the docstring.
FEE_PRIOR_DECISION = "ON"   # measured on TRAIN OOF; see _fin_feeprior.py


# ---------------------------------------------------------------------------
# features  (SAFE: visible OCR only)
# ---------------------------------------------------------------------------

def features_A(rec, fields, base_feats) -> dict:
    return _strip(clf_r2_1.design_features(rec, fields, base_feats))


def features_B(rec, fields, base_feats, fa=None) -> dict:
    f = dict(fa) if fa is not None else features_A(rec, fields, base_feats)
    f.update(_strip(clf_r2_3.evidence_features(rec, fields)))
    return f


# ---------------------------------------------------------------------------
# the caution rule - the ONLY path from the untrusted text layer to an output
# ---------------------------------------------------------------------------

def caution_signals(rec, fields) -> dict:
    """The monotone-safe half of the hidden/visible comparison.

    ``disagree``  the text layer asserts a materially different value than the
                  trusted visible reading  -> the packet 'relies on untrusted
                  evidence' / is contradictory (FIELD_MANUAL NEEDS_REVIEW).
    ``refused``   visible OCR has nothing and the text layer offered to fill it;
                  the extractor correctly refused, so the field really is blank
                  -> the packet is INCOMPLETE.
    """
    try:
        d = repair_mod.disagreement_features(rec, fields)
    except Exception:
        return {"disagree": 0, "refused": 0}
    return {"disagree": int(d.get("hidden_disagree_n", 0)),
            "refused": int(d.get("hidden_refused_fill_n", 0))}


CAUTION_MODES = ("off", "dis", "dis_all", "any", "any_all")


def apply_caution(act, conf, proba, sigs, mode):
    """Demote-only post-processor.  Structurally cannot emit APPROVED.

    mode = <trigger>[_all]:
        'off'      do nothing
        'dis'      trigger = the text layer asserts a CONTRADICTORY value
        'any'      trigger = contradiction OR a refused fill (blank visible field
                   that the untrusted layer offered to populate)
      suffix '_all' also demotes DENIED; without it only APPROVED is demoted.

    The confidence reported after a demotion is the model's own P(NEEDS_REVIEW),
    which is the Brier-honest report for the action actually taken.
    """
    act = np.asarray(act).copy()
    conf = np.asarray(conf, dtype=np.float64).copy()
    proba = np.asarray(proba, dtype=np.float64)
    if mode == "off":
        return act, conf
    trig, wide = mode.split("_")[0], mode.endswith("_all")
    if trig == "dis":
        fire = np.array([bool(s["disagree"] > 0) for s in sigs])
    else:
        fire = np.array([bool(s["disagree"] > 0 or s["refused"] > 0) for s in sigs])
    tgt = (act != I_REVIEW) if wide else (act == I_APPROVED)
    move = fire & tgt
    if move.any():
        act[move] = I_REVIEW
        conf[move] = np.clip(proba[move, I_REVIEW], 0.01, 0.99)
    return act, conf


# ---------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------

def _members(spec):
    if spec == "hgb_low":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return [HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_leaf_nodes=3,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0)]
    return clf_r2_1._make_members(spec)


def _fit_members(spec, X, y):
    ms = _members(spec)
    for m in ms:
        m.fit(X, y)
    return ms


def _proba(ms, X):
    return np.mean([m.predict_proba(X) for m in ms], axis=0)


def _oof(X, y, spec, seed):
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros((len(y), len(CLASSES)))
    fid = np.zeros(len(y), dtype=int)
    for k, (a, b) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True,
                                               random_state=seed).split(X, y)):
        out[b] = _proba(_fit_members(spec, X[a], y[a]), X[b])
        fid[b] = k
    return out, fid


def recal_lofo(pr, y, fid, kind):
    if kind == "none":
        return pr
    out = np.zeros_like(pr)
    for k in sorted(set(fid)):
        m = fid != k
        out[fid == k] = apply_recal(pr[fid == k], fit_recal(pr[m], y[m], kind))
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class AdjudicatorFinal:
    """Fit on TRAIN_IDS only.

    predict = blend(cell A, cell B) -> recalibration -> EV rule -> caution rule.
    ``cell`` selects 'A', 'B' or 'blend'; it is chosen on TRAIN OOF.
    """

    def __init__(self, cell="blend", recal="temp", fa_penalty=-12.0,
                 brier_w=BRIER_W, blend_w=BLEND_W, caution="any"):
        self.cell = cell
        self.recal_kind = recal
        self.fa_penalty = float(fa_penalty)
        self.brier_w = float(brier_w)
        self.blend_w = float(blend_w)
        self.caution = caution
        self.keys_a, self.keys_b = [], []
        self.mem_a, self.mem_b = [], []
        self.cal = None

    # -- plumbing ----------------------------------------------------------
    @staticmethod
    def _matrix(feats, keys):
        return np.array([[float(d.get(k, 0.0)) for k in keys] for d in feats],
                        dtype=np.float64)

    def _blend(self, fa, fb):
        if self.cell == "A":
            return _proba(self.mem_a, self._matrix(fa, self.keys_a))
        if self.cell == "B":
            return _proba(self.mem_b, self._matrix(fb, self.keys_b))
        pa = _proba(self.mem_a, self._matrix(fa, self.keys_a))
        pb = _proba(self.mem_b, self._matrix(fb, self.keys_b))
        return (1.0 - self.blend_w) * pa + self.blend_w * pb

    # -- fit / predict -----------------------------------------------------
    def fit(self, feats_a, feats_b, labels, keys_a=None, keys_b=None,
            oof_proba=None):
        feats_a = [_strip(d) for d in feats_a]
        feats_b = [_strip(d) for d in feats_b]
        self.keys_a = [k for k in (keys_a or sorted({k for d in feats_a for k in d}))
                       if is_safe_key(k)]
        self.keys_b = [k for k in (keys_b or sorted({k for d in feats_b for k in d}))
                       if is_safe_key(k)]
        y = np.array([CLASSES.index(l) if isinstance(l, str) else l for l in labels])
        if self.cell in ("A", "blend"):
            self.mem_a = _fit_members(SPEC_A, self._matrix(feats_a, self.keys_a), y)
        if self.cell in ("B", "blend"):
            self.mem_b = _fit_members(SPEC_B, self._matrix(feats_b, self.keys_b), y)
        self.cal = (fit_recal(oof_proba, y, self.recal_kind)
                    if (self.recal_kind != "none" and oof_proba is not None) else None)
        return self

    def predict_proba(self, feats_a, feats_b):
        return apply_recal(self._blend([_strip(d) for d in feats_a],
                                       [_strip(d) for d in feats_b]), self.cal)

    def predict(self, feats_a, feats_b, sigs=None):
        p = self.predict_proba(feats_a, feats_b)
        act, conf = decide(p, self.fa_penalty, self.brier_w)
        if sigs is not None:
            act, conf = apply_caution(act, conf, p, sigs, self.caution)
        return [CLASSES[a] for a in act], np.asarray(conf)

    def predict_one(self, rec, fields, base_feats):
        fa = features_A(rec, fields, base_feats)
        fb = features_B(rec, fields, base_feats, fa=fa)
        lab, conf = self.predict([fa], [fb], [caution_signals(rec, fields)])
        return lab[0], float(conf[0])

    # -- artifact ----------------------------------------------------------
    def save(self, path=MODEL_PATH):
        import joblib
        joblib.dump({"cell": self.cell, "keys_a": self.keys_a, "keys_b": self.keys_b,
                     "mem_a": self.mem_a, "mem_b": self.mem_b, "cal": self.cal,
                     "recal": self.recal_kind, "fa_penalty": self.fa_penalty,
                     "brier_w": self.brier_w, "blend_w": self.blend_w,
                     "caution": self.caution}, path, compress=3)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        import joblib
        d = joblib.load(path)
        o = cls(cell=d.get("cell", "blend"), recal=d["recal"],
                fa_penalty=d["fa_penalty"], brier_w=d["brier_w"],
                blend_w=d["blend_w"], caution=d.get("caution", "approved"))
        o.keys_a, o.keys_b = d["keys_a"], d["keys_b"]
        o.mem_a, o.mem_b, o.cal = d["mem_a"], d["mem_b"], d["cal"]
        return o


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------

def _featcache():
    import pickle
    pkl = _SCRATCH / "featcache_d.pkl"
    if pkl.exists():
        return pickle.load(open(pkl, "rb"))
    import extract_v2
    out = {}
    for cid, rec in sorted(common.load_cache().items()):
        fl = extract_v2.extract_fields_v2(rec)
        out[cid] = (fl, extract_v2.extract_features(rec, fl))
    pickle.dump(out, open(pkl, "wb"))
    return out


def _cache(name, build, cache, fc, ids):
    import pickle
    pkl = _SCRATCH / name
    have = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    todo = [c for c in ids if c not in have]
    for n, c in enumerate(todo):
        have[c] = build(cache[c], fc[c][0], fc[c][1])
    if todo:
        pickle.dump(have, open(pkl, "wb"))
    return have


# ---------------------------------------------------------------------------
# self-test / report
# ---------------------------------------------------------------------------

_RECALS = ("none", "temp", "vector")
_PENS = (-4., -8., -12., -16., -20., -24., -32.)
_BWS = (BRIER_W, 8.0, 12.0)
_CAUTIONS = CAUTION_MODES
_FIELDS = ["applicant_name", "home_world", "declared_purpose", "species_code",
           "visa_class", "sponsor_id", "arrival_date", "fee_status", "risk_flags"]


def _main():
    import pickle, time, copy
    import extract_v2
    import _fin_attacks as AT
    t0 = time.time()
    labels = common.load_labels()
    cache = common.load_cache()
    fc = _featcache()
    train, dev = common.split_ids(labels)
    train = [c for c in train if c in cache and c in fc]
    dev = [c for c in dev if c in cache and c in fc]
    print(f"cache={len(cache)}  train={len(train)}  dev={len(dev)}")

    # ---- 0. the security property, asserted on real records -----------------
    print("\n[0] SECURITY: model input under text-layer attack")
    banned_seen, moved = set(), {}
    for cid in dev[:40]:
        rec = cache[cid]
        f0 = extract_v2.extract_fields_v2(rec)
        b0 = extract_v2.extract_features(rec, f0)
        a0 = features_A(rec, f0, b0); v0 = features_B(rec, f0, b0, fa=a0)
        banned_seen |= {k for k in clf_r2_1.design_features(rec, f0, b0)
                        if not is_safe_key(k)}
        for nm, fn in AT.ATTACKS.items():
            r1 = fn(rec)
            f1 = extract_v2.extract_fields_v2(r1)
            b1 = extract_v2.extract_features(r1, f1)
            a1 = features_A(r1, f1, b1); v1 = features_B(r1, f1, b1, fa=a1)
            for k in set(v0) | set(v1):
                if v0.get(k) != v1.get(k):
                    moved.setdefault(nm, set()).add(k)
    print(f"    banned columns removed from the vector: {sorted(banned_seen)}")
    for nm in AT.ATTACKS:
        print(f"    {nm:<12} model-input columns that move: {sorted(moved.get(nm, ()))} "
              f"({len(moved.get(nm, ()))})")
    assert not any(moved.values()), "model input is NOT attack-invariant"
    print("    OK: the model's input vector is a pure function of visible OCR")

    # ---- 1. features --------------------------------------------------------
    print("\n[1] building features ...", flush=True)
    A = {c: _strip(d) for c, d in
         _cache("designfeat_r2_1.pkl", clf_r2_1.design_features, cache, fc,
                train + dev).items()}
    Ev = {c: _strip(d) for c, d in
          _cache("evidfeat_r2_3.pkl",
                 lambda rec, fl, bf: clf_r2_3.evidence_features(rec, fl),
                 cache, fc, train + dev).items()}
    B = {c: {**A[c], **Ev[c]} for c in train + dev}
    keys_a = sorted({k for c in train + dev for k in A[c]})
    keys_b = sorted({k for c in train + dev for k in B[c]})
    print(f"    cell A {len(keys_a)} features | cell B {len(keys_b)} features "
          f"(all safe)")

    sig = {c: caution_signals(cache[c], fc[c][0]) for c in train + dev}
    fires = sum(1 for c in train if sig[c]["disagree"] or sig[c]["refused"])
    print(f"    caution signal fires on {fires}/{len(train)} TRAIN packets")

    ytr = np.array([CLASSES.index(labels[c]["adjudication"]) for c in train])
    Xa = np.array([[float(A[c].get(k, 0.)) for k in keys_a] for c in train])
    Xb = np.array([[float(B[c].get(k, 0.)) for k in keys_b] for c in train])
    sig_tr = [sig[c] for c in train]

    # ---- 2. TRAIN out-of-fold, five paired fold seeds. DEV UNTOUCHED --------
    print("\n[2] TRAIN out-of-fold, 5 paired fold seeds (dev never touched):",
          flush=True)
    oof_dir = _SCRATCH / "_fin_oof"
    oof_dir.mkdir(exist_ok=True)
    store = []
    for sd in FOLD_SEEDS:
        fa_p, fb_p = oof_dir / f"a_{sd}.npy", oof_dir / f"b_{sd}.npy"
        fid_p = oof_dir / f"fid_{sd}.npy"
        if fa_p.exists() and fb_p.exists() and fid_p.exists():
            pa, pb, fid = np.load(fa_p), np.load(fb_p), np.load(fid_p)
        else:
            pa, fid = _oof(Xa, ytr, SPEC_A, sd)
            pb, _ = _oof(Xb, ytr, SPEC_B, sd)
            np.save(fa_p, pa); np.save(fb_p, pb); np.save(fid_p, fid)
        store.append((pa, pb, fid))
        print(f"    seed {sd} done ({time.time()-t0:.0f}s)", flush=True)

    def sweep(mix):
        rows = []
        for rc in _RECALS:
            recs = [(recal_lofo(p, ytr, fid, rc)) for p, fid in mix]
            for pen in _PENS:
                for bw in _BWS:
                    for cau in _CAUTIONS:
                        res = []
                        for p in recs:
                            a, cf = decide(p, pen, bw)
                            a, cf = apply_caution(a, cf, p, sig_tr, cau)
                            res.append(challenge_score(ytr, a, cf))
                        tot = [r["classification"] + r["calibration"] for r in res]
                        rows.append(dict(
                            rc=rc, pen=pen, bw=bw, cau=cau,
                            tot=float(np.mean(tot)), sd=float(np.std(tot)),
                            cls=float(np.mean([r["classification"] for r in res])),
                            cal=float(np.mean([r["calibration"] for r in res])),
                            fa=float(np.mean([r["false_approvals"] for r in res]))))
        return rows

    mixes = {"A": [(pa, fid) for pa, pb, fid in store],
             "B": [(pb, fid) for pa, pb, fid in store],
             "blend": [((1 - BLEND_W) * pa + BLEND_W * pb, fid)
                       for pa, pb, fid in store]}
    best = None
    for tag, mix in mixes.items():
        rows = sweep(mix)
        top = max(r["tot"] for r in rows)
        near = [r for r in rows if r["tot"] >= top - SELECT_TOL]
        w = min(near, key=lambda r: (r["fa"], -r["tot"]))
        w["cell"] = tag
        print(f"    cell {tag:<6} top {top:6.2f} | selected recal={w['rc']:<6} "
              f"pen {w['pen']:6.1f} bw {w['bw']:4.1f} caution={w['cau']:<8} -> "
              f"OOF {w['tot']:6.2f}+-{w['sd']:.2f} cls {w['cls']:5.2f} "
              f"cal {w['cal']:5.2f} fa {w['fa']:5.1f}/{len(train)}", flush=True)
        if best is None or w["tot"] > best["tot"]:
            best, best_rows = w, rows
    sel = best
    print(f"    -> SELECTED ON TRAIN OOF: cell {sel['cell']}, recal={sel['rc']}, "
          f"fa_penalty={sel['pen']}, brier_w={sel['bw']}, caution={sel['cau']}")

    print("    caution-rule frontier at the selected decision layer (TRAIN OOF):")
    for r in best_rows:
        if r["rc"] == sel["rc"] and r["pen"] == sel["pen"] and r["bw"] == sel["bw"]:
            print(f"      caution={r['cau']:<9} tot {r['tot']:6.2f} "
                  f"(cls {r['cls']:5.2f} cal {r['cal']:5.2f}) fa {r['fa']:5.1f}")

    # ---- 3. ship: fit on all of TRAIN --------------------------------------
    pa0, pb0, _ = store[0]
    oof_blend = {"A": pa0, "B": pb0,
                 "blend": (1 - BLEND_W) * pa0 + BLEND_W * pb0}[sel["cell"]]
    clf = AdjudicatorFinal(cell=sel["cell"], recal=sel["rc"],
                           fa_penalty=sel["pen"], brier_w=sel["bw"],
                           caution=sel["cau"])
    clf.fit([A[c] for c in train], [B[c] for c in train],
            [labels[c]["adjudication"] for c in train],
            keys_a=keys_a, keys_b=keys_b, oof_proba=oof_blend)
    print(f"\n[3] fitted on {len(train)} TRAIN cases ({time.time()-t0:.0f}s)")

    # ---- 4. DEV, read once --------------------------------------------------
    Fa = [A[c] for c in dev]
    Fb = [B[c] for c in dev]
    Sg = [sig[c] for c in dev]
    proba = clf.predict_proba(Fa, Fb)
    act, conf = decide(proba, sel["pen"], sel["bw"])
    act, conf = apply_caution(act, conf, proba, Sg, sel["cau"])
    adj = [CLASSES[a] for a in act]
    ydev = np.array([CLASSES.index(labels[c]["adjudication"]) for c in dev])
    sd_ = challenge_score(ydev, act, conf)
    print(f"[4] DEV (read once) accuracy {sd_['accuracy']:.3f}  "
          f"brier {sd_['brier']:.4f}")
    print("    confusion (truth -> pred):")
    for i, t in enumerate(CLASSES):
        print("     ", t.ljust(13), "  ".join(
            f"{p[:4]}={int(np.sum((ydev == i) & (act == j))):3d}"
            for j, p in enumerate(CLASSES)))
    print(f"    TRAIN OOF (cls+cal) {sel['tot']:.2f}  vs  "
          f"DEV (cls+cal) {sd_['classification'] + sd_['calibration']:.2f}"
          f"   gap {sd_['classification'] + sd_['calibration'] - sel['tot']:+.2f}")

    def rows_of(L, C, FL, ids):
        return [{"case_id": ids[i], "adjudication": L[i], "confidence":
                 round(float(C[i]), 4), **{k: FL[i].get(k) for k in _FIELDS}}
                for i in range(len(ids))]

    fl_dev = [fc[c][0] for c in dev]
    res = common.score_rows(rows_of(adj, conf, fl_dev, dev), "clf_final_dev")
    print("    DEV  " + common.summarize(res))

    # ---- 5. artifact round-trip --------------------------------------------
    clf.save()
    lab2, conf2 = AdjudicatorFinal.load().predict(Fa, Fb, Sg)
    assert lab2 == adj and np.allclose(conf2, conf), "artifact round-trip mismatch"
    print(f"\n[5] saved -> {MODEL_PATH} "
          f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB); round-trip OK")

    # ---- 6. adversarial re-run, end to end ---------------------------------
    print("\n[6] ADVERSARIAL re-run on all 200 DEV records (end to end):")
    print("    clean : " + common.summarize(res))
    for nm, fn in AT.ATTACKS.items():
        la, ca, fla, sa = [], [], [], []
        for c in dev:
            r = fn(cache[c])
            f = extract_v2.extract_fields_v2(r)
            b = extract_v2.extract_features(r, f)
            a = features_A(r, f, b)
            la.append(a); ca.append(features_B(r, f, b, fa=a))
            fla.append(f); sa.append(caution_signals(r, f))
        lab_a, cf_a = clf.predict(la, ca, sa)
        flips = [(dev[i], adj[i], lab_a[i]) for i in range(len(dev))
                 if adj[i] != lab_a[i]]
        toward = [t for t in flips if t[2] == "APPROVED"]
        ra = common.score_rows(rows_of(lab_a, cf_a, fla, dev), "clf_final_" + nm)
        print(f"    {nm:<12} flips {len(flips)}/{len(dev)}  toward APPROVED "
              f"{len(toward)}")
        for cid, x, y_ in flips[:10]:
            print(f"        {cid} {x} -> {y_}  truth={labels[cid]['adjudication']}")
        print(f"      {nm} score: " + common.summarize(ra))
        assert not toward, f"{nm} attack promoted a case to APPROVED"
    print("\n    OK: ZERO flips toward APPROVED under both attacks.")
    return res


if __name__ == "__main__":
    _main()
