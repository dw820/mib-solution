"""ROUND 2 adjudication classifier: DESIGN D + a corrected output layer.

Starts from ``clf_d`` (adjudicator-note reader + policy features + expected-value
decision rule, dev 121.34/150) and changes three things, each chosen by TRAIN
out-of-fold score with DEV touched exactly once at report time.

WHAT THE ERROR ANALYSIS FOUND
-----------------------------
Points lost on dev by clf_d, by bucket (max 8/case, 200 cases = 1600 raw):

    truth DENIED   -> pred APPROVED      10 cases x 12 =  120   <-- biggest
    truth DENIED   -> pred NEEDS_REVIEW  12 cases x  6 =   72
    truth APPROVED -> pred DENIED         8 cases x  8 =   64
    truth NR       -> pred DENIED         6 cases x  7 =   42
    truth APPROVED -> pred NEEDS_REVIEW   5 cases x  6 =   30
    truth NR       -> pred APPROVED       2 cases x  7 =   14

So the dominant bucket is *true DENIED that we fail to deny* (22 cases, 192 of
the 342 raw points lost).  clf_d's own hypothesis for this was missed risk
flags, so that is what I attacked first.

ATTACK 1 (FAILED, and it is worth saying so).  The risk-flag detector in
``flags.py`` only reads flag names that follow an "Observed flags:" anchor on a
biometric slip, or a "Reason: ... risk flag" clause in an adjudicator note.  It
is 99.6% precise but only 59.6% recall on TRAIN, so the obvious move is an
UNANCHORED fuzzy scan for the eight canonical flag names anywhere in visible
OCR.  I built it and measured it on TRAIN:

    unanchored whole-document scan, threshold 0.84:  precision 1.000  recall 0.534
    existing anchored detector:                      precision 0.996  recall 0.596

The unanchored scan is *worse*, and it finds exactly ONE case the anchored
detector misses.  The missing 40% of flags are simply not printed anywhere in
visible OCR - they were cut out, washed out, or live only in hidden text.  There
is no text-side recall left to win, and this block is therefore absent from the
final model.  (Reproduce: ``scratchpad/probe_flagrecall.py``.)

What that leaves is a bucket of 350/800 TRAIN packets carrying *no* visible
policy evidence at all - no note, no flag mention, no transit/unpaid/revoked/
embargo condition - whose label distribution is A .48 / D .25 / NR .27.  On
those the visible evidence genuinely does not determine the answer, so the only
honest way to earn more points there is to be better at *knowing that*, i.e. a
better probability and a decision rule that spends the uncertainty correctly.
That is what the three changes below do.

THE THREE CHANGES
-----------------
1.  **Calibration-aware decision rule.**  clf_d maximises expected
    classification points only.  But the challenge total is
    ``classification + calibration``, and with N cases

        total = (10/N) * sum_i [ raw_i - 4 * brier_i ] + 20

    so one unit of Brier error is worth exactly 4 raw classification points.
    For action ``a`` the optimal confidence is ``p_a = P(a)`` (it minimises the
    Brier term), and at that confidence ``E[brier] = p_a (1 - p_a)``.  The
    action to take is therefore

        argmax_a  E[raw | a] - BRIER_W * p_a * (1 - p_a),   BRIER_W = 4

    which is the same expected-value rule plus an explicit dislike of acting on
    a near-coin-flip.  ``BRIER_W`` is a derived constant, not a tuned knob; the
    selector still checks 0 against it on TRAIN OOF.

2.  **Leave-one-fold-out probability recalibration.**  A temperature or diagonal
    matrix-scaling map is fitted on TRAIN out-of-fold log-probabilities, with
    the map applied to fold k fitted only on the other folds.  Boosted trees on
    800 rows are over-sharp, and the decision rule above is only as good as the
    probabilities fed to it.  1-6 parameters, no extra model fits.

3.  **A genuinely heterogeneous ensemble**, plus two small new feature blocks.
    Note that seed-ensembling the HistGradientBoostingClassifier is a *no-op*:
    at fixed ``max_iter``, with no early stopping and no subsampling, it is
    deterministic, and 1 vs 3 seeds gave bit-identical TRAIN OOF scores.  The
    variance reduction has to come from different learners, so the ensemble
    averages a boosted tree, two randomized-tree forests and a regularized
    multinomial logistic model.  The two new feature blocks are a standalone
    visible adjudicator STAMP reader (FIELD_MANUAL precedence rank 1: an
    isolated short line that *is* the word DENIED/APPROVED, as opposed to
    clf_d's document-level bag-of-words bit, which also fires on the SAMPLE
    DENIAL watermark and on injected barcode payloads) and a per-page-type
    legibility profile (lines and mean OCR confidence per page type), which
    tells the model *which* evidence source is damaged rather than only that
    the packet as a whole is.

MEASURED RESULTS (DEV_IDS = 200 held-out cases, real ``scripts/evaluate.py``)
----------------------------------------------------------------------------
                        total    class   extract   calib   false-approvals
    prior baseline      74.27    41.62    43.92     8.77          0
    clf_d (round 1)    121.34    62.90    43.92    14.53         10
    this module        123.16    63.15    43.92    16.09          4

Extraction is untouched at 43.92/50; the whole delta is decisions.  dev
accuracy 0.755, Brier 0.0978 (clf_d 0.1368).  Note the *shape* of the gain:
classification is nearly flat (+0.25) while calibration is up 1.56 and
catastrophic false approvals are down from 10 to 4.  That is exactly what the
calibration-aware rule plus the OOF-selected penalty are supposed to buy - the
approvals given up were near-coin-flips worth almost nothing in expectation.

Confusion (truth -> pred):
    APPROVED       A 28   D  7   NR 16
    DENIED         A  4   D 73   NR 17
    NEEDS_REVIEW   A  2   D  3   NR 50

Where the TRAIN out-of-fold score comes from (repeated over 3 fold seeds,
decision fixed at recal=temp / penalty -4 / brier_w 4, cls+cal out of 100):

    clf_d feature vector only ................... 76.86 +- 0.36   (fa 29.0)
    + stamp reader .............................. 77.67 +- 0.66   (fa 25.3)
    + per-page-type legibility profile .......... 78.17 +- 0.31   (fa 24.3)
    + both ...................................... 77.84 +- 0.44   (fa 25.7)

Both blocks help individually; combining them is not additive and "page alone"
edges out "both" by 0.33, which is within one standard deviation.  I kept both
anyway: the stamp reader is precedence-rank-1 evidence and should transfer to
private layouts better than clf_d's trap-prone document-level token bit, and
picking the single best of four noisy TRAIN numbers is itself an overfit.  This
is a judgement call and it is worth stating rather than hiding.

Ensemble composition, same protocol (2 fold seeds, feature set = both):

    hgb only ............ 77.92    hgb+ET2 ......... 78.96
    ET only ............. 77.42    hgb+ET2+RF2 ..... 78.52
                                   hgb+ET2+LR ...... 79.11   <- shipped family

HONEST LIMITS
-------------
*   The improvement is real but small in classification terms.  56% of TRAIN
    packets carry no visible evidence that determines the decision, and on
    those the model is near the base rate.  That ceiling was not moved and I do
    not believe it can be moved from the text side (see ATTACK 1).
*   DEV was scored twice with this module: once at 122.13 with an earlier
    ensemble grid, once at 123.16 final.  Nothing was selected using DEV - the
    grid change was driven entirely by the TRAIN OOF sweep in
    ``scratchpad/exp_ens.py`` - but two looks is two looks, so treat the
    reported 123.16 as having roughly a half-point of optimism.
*   ``vector`` (diagonal matrix scaling) won the recalibration slot on TRAIN by
    a hair over ``temp``.  It has 6 parameters instead of 1 and is therefore
    the more fragile of the two on an unseen distribution; if the private set
    looks different, ``temp`` is the safer swap and costs ~0.2 OOF points.

COMPLIANCE
----------
*   Nothing is keyed to a case_id; the only literal identifiers are the three
    revoked sponsors printed in the public FIELD_MANUAL (inherited from clf_d).
*   Model, recalibrator and every threshold are fitted on TRAIN_IDS only.
*   Only visible OCR is read.  hidden_text never supplies a value; it reaches
    the model solely through extract_v2's disagreement counts.
*   Every added signal is additive: an unseen private layout simply zeroes them
    and the model falls back on document-shape and resolution features.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common          # noqa: E402
import clf_d           # noqa: E402  (note reader + policy features are reused)

CLASSES = clf_d.CLASSES
MODEL_PATH = _HERE / "clf_r2_1_model.joblib"

# 1 unit of Brier error == 4 raw classification points (see module docstring).
BRIER_W = 4.0

_PAGE_TYPES = ("note", "intake", "fee", "slip", "sponsor", "registry", "other")
_STAMP_WORDS = {"denied": "DENIED", "approved": "APPROVED",
                "needsreview": "NEEDS_REVIEW"}
_STAMP_THR = 0.80          # fitted on TRAIN only


def _sq(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# ---------------------------------------------------------------------------
# new feature block A: standalone visible adjudicator stamp (precedence rank 1)
# ---------------------------------------------------------------------------

def stamp_features(rec) -> dict:
    """An isolated short line that *is* a decision word.

    clf_d already has document-level ``tok_denied`` / ``tok_approved`` bits, but
    those fire on any occurrence of the word - including the documented
    "sample denial" watermark trap and injected barcode payloads.  A stamp is a
    short standalone line, so requiring the whole line to fuzzy-equal the word
    is a much stronger piece of rank-1 evidence.  Graded ratios are exposed as
    well as the boolean, so the model can discount marginal reads itself.
    """
    best = {v: 0.0 for v in _STAMP_WORDS.values()}
    big_denied = 0
    for l in common.doc_lines(rec):
        s = _sq(l["t"])
        if not (4 <= len(s) <= 14):
            continue
        rel_h = (l["y1"] - l["y0"]) / max(l["ph"], 1)
        for w, dec in _STAMP_WORDS.items():
            r = difflib.SequenceMatcher(None, w, s).ratio()
            if r > best[dec]:
                best[dec] = r
                if dec == "DENIED" and r >= _STAMP_THR and rel_h > 0.02:
                    big_denied = 1
    out = {f"stamp_r_{k}": v for k, v in best.items()}
    top = max(best, key=best.get)
    out["stamp_any"] = int(best[top] >= _STAMP_THR)
    for k in _STAMP_WORDS.values():
        out[f"stamp_is_{k}"] = int(best[top] >= _STAMP_THR and top == k)
    out["stamp_big_denied"] = big_denied
    return out


# ---------------------------------------------------------------------------
# new feature block B: per-page-type legibility profile
# ---------------------------------------------------------------------------

def page_profile(rec, fields) -> dict:
    """Which evidence source is damaged, not just how damaged the packet is.

    extract_v2 exposes ``has_page_<t>`` booleans and document-level confidence
    percentiles.  A packet whose *biometric slip* is unreadable and one whose
    *fee receipt* is unreadable have very different priors, and that is
    invisible to a document-level average.
    """
    meta = (fields or {}).get("_meta", {})
    pts = meta.get("page_types", [])
    agg = {t: [0, 0.0, 0] for t in _PAGE_TYPES}
    for i, p in enumerate(common.pages_of(rec)):
        t = pts[i] if i < len(pts) else "other"
        if t not in agg:
            t = "other"
        ls = common.page_lines(p)
        agg[t][0] += len(ls)
        agg[t][1] += sum(float(l.get("s", 0.0)) for l in ls)
        agg[t][2] += 1
    out = {}
    for t in _PAGE_TYPES:
        n, sc, npg = agg[t]
        out[f"pt_lines_{t}"] = float(n)
        out[f"pt_conf_{t}"] = float(sc / n) if n else 0.0
        out[f"pt_npages_{t}"] = float(npg)
    present = [out[f"pt_conf_{t}"] for t in _PAGE_TYPES if out[f"pt_npages_{t}"]]
    out["pt_conf_worst"] = min(present) if present else 0.0
    out["pt_conf_spread"] = (max(present) - min(present)) if present else 0.0
    out["pt_n_missing"] = float(sum(1 for t in _PAGE_TYPES if t != "other"
                                    and not out[f"pt_npages_{t}"]))
    return out


def design_features(rec, fields, base_feats) -> dict:
    """clf_d's vector + the two round-2 blocks. base_feats = the stock 97."""
    f = clf_d.design_features(rec, fields, base_feats)
    f.update(stamp_features(rec))
    f.update(page_profile(rec, fields))
    return f


# ---------------------------------------------------------------------------
# probability recalibration (temperature / diagonal matrix scaling)
# ---------------------------------------------------------------------------

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_recal(p, y, kind):
    """kind: 'none' | 'temp' (1 param) | 'vector' (2K params). Fit by NLL."""
    if kind == "none":
        return None
    from scipy.optimize import minimize
    lp = np.log(np.clip(p, 1e-9, 1.0))
    n, k = p.shape
    idx = np.arange(n)

    def nll(par):
        z = lp / np.exp(par[0]) if kind == "temp" else lp * np.exp(par[:k]) + par[k:]
        q = _softmax(z)
        return -np.mean(np.log(np.clip(q[idx, y], 1e-12, 1.0)))

    x0 = np.zeros(1 if kind == "temp" else 2 * k)
    r = minimize(nll, x0, method="Nelder-Mead",
                 options={"maxiter": 4000, "fatol": 1e-7, "xatol": 1e-5})
    return (kind, r.x)


def apply_recal(p, cal):
    if cal is None:
        return p
    kind, par = cal
    lp = np.log(np.clip(p, 1e-9, 1.0))
    k = p.shape[1]
    z = lp / np.exp(par[0]) if kind == "temp" else lp * np.exp(par[:k]) + par[k:]
    return _softmax(z)


# ---------------------------------------------------------------------------
# calibration-aware expected-value decision rule
# ---------------------------------------------------------------------------

def payoff(fa_penalty: float = -4.0) -> np.ndarray:
    """rows = action, cols = truth (APPROVED, DENIED, NEEDS_REVIEW)."""
    return np.array([
        [8.0, fa_penalty, 1.0],
        [0.0,        8.0, 1.0],
        [2.0,        2.0, 8.0],
    ])


def decide(proba, fa_penalty=-4.0, brier_w=BRIER_W):
    """Maximise E[raw classification points] - brier_w * E[Brier error].

    Confidence is P(chosen class): that is both the Brier-optimal report and,
    by construction, the probability the decision is correct.
    """
    proba = np.asarray(proba, dtype=np.float64)
    ev = proba @ payoff(fa_penalty).T - brier_w * proba * (1.0 - proba)
    act = ev.argmax(1)
    conf = proba[np.arange(len(proba)), act]
    return act, np.clip(conf, 0.01, 0.99)


def challenge_score(y, act, conf):
    """EVALUATION.md classification + calibration terms (verified vs evaluate.py)."""
    y = np.asarray(y)
    act = np.asarray(act)
    raw = 0
    fa = 0
    for t, p in zip(y, act):
        if t == p:
            raw += 8
        elif p == 2:
            raw += 2
        elif t == 2:
            raw += 1
        elif t == 1 and p == 0:
            raw -= 4
            fa += 1
    brier = float(np.mean([(c - float(t == p)) ** 2 for t, p, c in zip(y, act, conf)]))
    return {"classification": 80.0 * raw / (8.0 * len(y)),
            "calibration": 20.0 * max(0.0, 1.0 - 2.0 * brier),
            "false_approvals": fa,
            "accuracy": float(np.mean(act == y)),
            "brier": brier}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

# selected on TRAIN out-of-fold (see __main__); DEV played no part in this
DEFAULT_SPEC = "hgb+ET2+LR"
DEFAULT_RECAL = "vector"
DEFAULT_FA_PENALTY = -16.0


def _make_members(spec):
    """Ensemble members named by a '+'-joined spec string.

    Seed-ensembling a HistGradientBoostingClassifier is a *no-op*: at a fixed
    ``max_iter`` with no early stopping and no subsampling it is deterministic,
    so ``random_state`` changes nothing.  (Measured: 1 vs 3 seeds gave
    bit-identical TRAIN OOF scores.)  Variance reduction therefore has to come
    from genuinely different learners, which is what this ensemble is: a boosted
    tree, randomized trees, and a linear model, averaged.
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    parts = set(spec.split("+"))
    ms = []
    if "hgb" in parts:
        ms.append(HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0))
    if "ET2" in parts:
        ms += [ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                    random_state=100 + i, n_jobs=2)
               for i in range(2)]
    if "LR" in parts:
        ms.append(make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.3, max_iter=2000)))
    if not ms:
        raise ValueError(f"empty ensemble spec {spec!r}")
    return ms


class Adjudicator2:
    """Fit on TRAIN_IDS only. Predict = recalibrated ensemble + EV decision."""

    def __init__(self, spec=None, recal=DEFAULT_RECAL,
                 fa_penalty=DEFAULT_FA_PENALTY, brier_w=BRIER_W):
        self.spec = spec or DEFAULT_SPEC
        self.recal_kind = recal
        self.fa_penalty = fa_penalty
        self.brier_w = brier_w
        self.keys = []
        self.members = []
        self.cal = None

    def _matrix(self, feats):
        return np.array([[float(d.get(k, 0.0)) for k in self.keys] for d in feats],
                        dtype=np.float64)

    def _raw_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def fit(self, feats, labels, keys=None, oof_proba=None):
        """oof_proba: TRAIN out-of-fold probabilities used to fit the
        recalibrator. Required for recal != 'none' so the map is not fitted on
        the model's own in-sample (over-confident) output."""
        self.keys = keys or sorted({k for d in feats for k in d})
        X = self._matrix(feats)
        y = np.array([CLASSES.index(l) if isinstance(l, str) else l for l in labels])
        self.members = _make_members(self.spec)
        for m in self.members:
            m.fit(X, y)
        self.cal = (fit_recal(oof_proba, y, self.recal_kind)
                    if (self.recal_kind != "none" and oof_proba is not None) else None)
        return self

    def predict_proba(self, feats):
        return apply_recal(self._raw_proba(self._matrix(feats)), self.cal)

    def predict(self, feats):
        p = self.predict_proba(feats)
        act, conf = decide(p, self.fa_penalty, self.brier_w)
        return [CLASSES[a] for a in act], conf

    def predict_one(self, rec, fields, base_feats):
        lab, conf = self.predict([design_features(rec, fields, base_feats)])
        return lab[0], float(conf[0])

    def save(self, path=MODEL_PATH):
        import joblib
        joblib.dump({"keys": self.keys, "members": self.members, "cal": self.cal,
                     "spec": self.spec, "recal": self.recal_kind,
                     "fa_penalty": self.fa_penalty, "brier_w": self.brier_w},
                    path, compress=3)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        import joblib
        d = joblib.load(path)
        o = cls(spec=d["spec"], recal=d["recal"], fa_penalty=d["fa_penalty"],
                brier_w=d["brier_w"])
        o.keys, o.members, o.cal = d["keys"], d["members"], d["cal"]
        return o


# ---------------------------------------------------------------------------
# TRAIN out-of-fold machinery (model selection; DEV is never used for fitting)
# ---------------------------------------------------------------------------

def raw_oof(X, y, spec, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros((len(y), len(CLASSES)))
    fid = np.zeros(len(y), dtype=int)
    for k, (a, b) in enumerate(StratifiedKFold(folds, shuffle=True,
                                               random_state=seed).split(X, y)):
        ms = _make_members(spec)
        for m in ms:
            m.fit(X[a], y[a])
        out[b] = np.mean([m.predict_proba(X[b]) for m in ms], axis=0)
        fid[b] = k
    return out, fid


def recal_oof(pr, y, fid, kind):
    """Leave-one-fold-out recalibration, so the reported OOF score is honest."""
    if kind == "none":
        return pr
    out = np.zeros_like(pr)
    for k in sorted(set(fid)):
        m = fid != k
        out[fid == k] = apply_recal(pr[fid == k], fit_recal(pr[m], y[m], kind))
    return out


# ---------------------------------------------------------------------------
# self-test / report
# ---------------------------------------------------------------------------

_SCRATCH = _HERE.parent


def _featcache():
    import pickle
    pkl = _SCRATCH / "featcache_d.pkl"
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


def _designcache(cache, fc, ids):
    import pickle
    pkl = _SCRATCH / "designfeat_r2_1.pkl"
    have = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    todo = [c for c in ids if c not in have]
    for n, c in enumerate(todo):
        have[c] = design_features(cache[c], fc[c][0], fc[c][1])
        if n and n % 250 == 0:
            print(f"    ... design features {n}/{len(todo)}", flush=True)
    if todo:
        pickle.dump(have, open(pkl, "wb"))
    return have


# candidates; the winner is picked by TRAIN out-of-fold score only
_SPECS = ["hgb", "hgb+ET2", "hgb+ET2+LR"]
_RECALS = ("none", "temp", "vector")
_PENS = (-4.0, -8.0, -16.0)
_BWS = (0.0, BRIER_W)

# Selection rule, declared before DEV was ever scored: maximise TRAIN OOF
# (classification + calibration); then, among every configuration within
# SELECT_TOL points of that maximum, take the one with the fewest catastrophic
# false approvals.  This mirrors EVALUATION.md, whose second tie-breaker is the
# false-approval count and whose minimum bar excludes a "false-approval
# pattern"; the OOF frontier prices the trade at well under half a point.
SELECT_TOL = 0.5


def _main():
    import pickle
    labels = common.load_labels()
    cache = common.load_cache()
    fc = _featcache()
    train, dev = common.split_ids(labels)
    train = [c for c in train if c in cache and c in fc]
    dev = [c for c in dev if c in cache and c in fc]
    print(f"cache={len(cache)}  train={len(train)}  dev={len(dev)}")

    print("  building design features ...", flush=True)
    D = _designcache(cache, fc, train + dev)
    keys = sorted({k for c in train + dev for k in D[c]})
    Xtr = np.array([[float(D[c].get(k, 0.0)) for k in keys] for c in train])
    ytr = np.array([CLASSES.index(labels[c]["adjudication"]) for c in train])
    n_new = sum(1 for k in keys if k.startswith(("stamp_", "pt_")))
    print(f"  {len(keys)} features ({n_new} new in round 2)")

    # ---- TRAIN out-of-fold selection; DEV untouched
    print("  TRAIN out-of-fold selection (dev never touched):")
    cands = []
    oof_by_spec = {}
    for spec in _SPECS:
        pr, fid = raw_oof(Xtr, ytr, spec)
        oof_by_spec[spec] = pr
        for rc in _RECALS:
            p = recal_oof(pr, ytr, fid, rc)
            for pen in _PENS:
                for bw in _BWS:
                    act, conf = decide(p, pen, bw)
                    s = challenge_score(ytr, act, conf)
                    cands.append((s["classification"] + s["calibration"],
                                  spec, rc, pen, bw, s))
            act, conf = decide(p, -4.0, BRIER_W)
            s = challenge_score(ytr, act, conf)
            print(f"    {spec:12s} recal={rc:6s} | ev(-4,bw4) "
                  f"{s['classification'] + s['calibration']:6.2f} "
                  f"(cls {s['classification']:.2f} cal {s['calibration']:.2f} "
                  f"fa {s['false_approvals']:3d} acc {s['accuracy']:.3f})", flush=True)

    top = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= top - SELECT_TOL]
    tot, spec, rc, pen, bw, s = min(near, key=lambda c: (c[5]["false_approvals"], -c[0]))
    print(f"  TRAIN OOF max {top:.2f}; {len(near)} configs within {SELECT_TOL}")
    print(f"  -> selected {spec}, recal={rc}, fa_penalty={pen}, brier_w={bw} "
          f"(TRAIN OOF cls+cal {tot:.2f}, fa {s['false_approvals']}/{len(train)})")

    # false-approval frontier of the winning model+recal, so the trade is visible
    print("  TRAIN OOF false-approval frontier (winning model, brier_w=4):")
    for c in sorted((c for c in cands
                     if c[1] == spec and c[2] == rc and c[4] == BRIER_W),
                    key=lambda c: -c[3]):
        print(f"      fa_penalty {c[3]:6.1f} -> sum {c[0]:6.2f} "
              f"(cls {c[5]['classification']:.2f} cal {c[5]['calibration']:.2f}) "
              f"false_approvals {c[5]['false_approvals']:3d}")

    # the recalibrator shipped in the artifact is fitted on the winning spec's
    # TRAIN out-of-fold probabilities, never on the model's in-sample output
    pr = oof_by_spec[spec]
    clf = Adjudicator2(spec=spec, recal=rc, fa_penalty=pen, brier_w=bw)
    clf.fit([D[c] for c in train], [labels[c]["adjudication"] for c in train],
            keys=keys, oof_proba=pr)
    print(f"  fitted on {len(train)} TRAIN cases, {len(clf.keys)} features")

    # ---- DEV report (measured once)
    Fdev = [D[c] for c in dev]
    proba = clf.predict_proba(Fdev)
    act, conf = decide(proba, pen, bw)
    adj = [CLASSES[a] for a in act]
    truth = [labels[c]["adjudication"] for c in dev]
    ydev = np.array([CLASSES.index(t) for t in truth])
    sd = challenge_score(ydev, act, conf)
    print(f"  dev accuracy {sd['accuracy']:.3f}  brier {sd['brier']:.4f}")
    print("  dev confusion (truth -> pred):")
    for i, t in enumerate(CLASSES):
        print("   ", t.ljust(13), "  ".join(
            f"{p[:4]}={int(np.sum((ydev == i) & (act == j))):3d}"
            for j, p in enumerate(CLASSES)))

    rows = []
    for c, a, cf in zip(dev, adj, conf):
        r = {"case_id": c, **{k: v for k, v in fc[c][0].items() if k != "_meta"}}
        r["adjudication"] = a
        r["confidence"] = round(float(cf), 4)
        rows.append(r)
    res = common.score_rows(rows, "clf_r2_1_dev")
    print("  DEV  " + common.summarize(res))

    clf.save()
    print(f"  saved -> {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # round-trip check: the shipped artifact reproduces the reported decisions
    lab2, conf2 = Adjudicator2.load().predict(Fdev)
    assert lab2 == adj and np.allclose(conf2, conf), "artifact round-trip mismatch"
    print("  artifact round-trip OK")
    return res


if __name__ == "__main__":
    _main()
