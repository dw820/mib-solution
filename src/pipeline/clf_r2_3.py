"""R2-3 - adjudication classifier attacked from the EVIDENCE side.

Design D (clf_d.py, dev 121.34) added an adjudicator-note reader, a handful of
FIELD_MANUAL policy bits, and an expected-value output layer.  Its own honest
limit section says where the remaining points are:

    "70% of packets carry no adjudicator note, and on those the model is only
     65.6% accurate ... Substituting ground-truth fields for the extractor's
     raises note-less accuracy to 0.869, so the remaining headroom lives in
     extraction (chiefly risk-flag recall), not here."

That diagnosis is right about *where* and wrong about *what to do*.  The
classifier does not need the extractor to output the correct value; it needs to
know **how the packet's own documents relate to one another**.  A packet whose
intake form says ``Applicant: Qortari Ix`` while its registry extract and
sponsor letter both say ``Luzarn Qortari`` is an ``identity_conflict`` - and it
is one *whether or not* the resolver picks the right name.  The stock 97-vector
collapses all of that into a single ``name_agreement`` scalar and a per-field
``nev_`` count; every other cross-document contradiction is invisible.

So this module keeps design D's note reader, policy bits and EV rule verbatim
(imported, not reimplemented) and adds five new families of *evidence*
features, all derived from visible OCR geometry and the extractor's own
candidate pool - never from hidden text.

  F1  CROSS-SOURCE AGREEMENT MATRIX.  ``extract_v2._collect_candidates`` already
      returns, per field, every candidate reading tagged with the page type that
      produced it.  The resolver throws that structure away and keeps one value.
      Here it is kept: per field, how many independent page types supplied a
      value, how many distinct values they formed, what share of the consensus
      weight the winner holds, and the explicit pairwise verdicts
      intake-vs-registry, intake-vs-slip, intake-vs-sponsor.  This is a direct
      observation of identity_conflict / sponsor_mismatch rather than an
      inference from the resolved output.

  F2  PACKET COMPLETENESS AND DAMAGE.  Every page type has a fixed label
      inventory; a torn or washed-out page shows up as missing labels, bracket
      placeholders (``[DATE WASHED OUT]``, ``[SPECIES WHITEOUT]``), blank pages,
      duplicated or foreign pages, and a low-confidence line tail.  "Incomplete
      or illegible" is one of the three NEEDS_REVIEW definitions in
      FIELD_MANUAL, and nothing in the stock vector measures it structurally.

  F3  STAMP GEOMETRY vs WATERMARK.  FIELD_MANUAL trap #1: "a watermark reading
      'sample denial' is not a denial".  A watermark and a stamp are the same
      *words*; they differ in geometry.  The watermark repeats on most pages at
      a large glyph height with mediocre OCR confidence; a real stamp appears
      once, usually on the note page.  Every decision-ish token therefore gets
      page-count, page-fraction, relative glyph height, OCR confidence and
      "is it on the adjudicator-note page" instead of one bag-of-words bit.

  F4  SPONSOR-ATTESTATION SEMANTIC CHECK.  The attestation letter is prose, so
      the key/value harvester never reads it, yet it restates the sponsor id,
      the applicant name, the case id and the visa class.  Parsing it gives four
      independent agreement bits - the cleanest sponsor_mismatch evidence a
      packet contains.

  F5  FEE-RECEIPT ARITHMETIC.  ``FeeStatus: paid`` with ``Amount: $0.00``, or
      ``waived`` with a positive amount and ``WaiverCode: N/A``, is a
      self-contradictory receipt.  The stock ``fee_src_*`` features say where the
      fee came from but never whether the receipt is internally consistent.

Everything else - model class, calibration, EV decision rule, TRAIN-only
selection protocol - is unchanged from design D on purpose, so the measured
delta is attributable to the features.

MEASURED RESULTS (DEV_IDS = 200 held-out cases, scripts/evaluate.py)
-------------------------------------------------------------------
                        total    class   calib   false-approvals   artifact
    prior baseline      74.27    41.62    8.77         0
    design D (clf_d)   121.34    62.90   14.53        10            4.8 MB
    this module        123.09    63.15   16.03         6            0.1 MB
    (extraction is unchanged at 43.92/50 throughout; all delta is decisions.)

  dev accuracy 0.800 by argmax, 0.770 under the EV rule (the EV rule trades
  accuracy for points on purpose).  Confusion (truth -> pred):
      APPROVED       A 30   D  9   NR 12
      DENIED         A  6   D 76   NR 12
      NEEDS_REVIEW   A  2   D  5   NR 48

WHERE THE +1.75 ACTUALLY CAME FROM - AND THE HONEST SURPRISE
------------------------------------------------------------
Not from the features.  Ablation on TRAIN out-of-fold, averaged over five
*paired* fold seeds (cls+cal, max 100):

    design-D features at design-D's capacity (15 leaves) ... 76.10   <- incumbent
    design-D features at 2-3 leaves ........................ 78.03   (+1.93)
    design-D + all evidence features at 3 leaves ........... 78.49   (+0.46)

Design D chose ``max_leaf_nodes=15`` on a **single** 5-fold split.  The per-seed
standard deviation of that metric is about 0.5 points - the same size as the
effects being compared - so the choice was noise-driven, and 15 leaves is well
past the overfitting point for 800 rows.  Almost the entire gain in this module
is that one correction, which is a model-head finding, not a feature finding.
The dev evidence agrees with the TRAIN diagnosis: the calibration term moved
14.53 -> 16.03 (a lower-capacity booster is simply less overconfident) while
classification barely moved, 62.90 -> 63.15.

The evidence features are worth a real but modest +0.46, and only at the right
capacity.  Their first measurement was NEGATIVE:

    at 15 leaves, design-D + all evidence features ......... 75.33   (-0.77)
    at 15 leaves, design-D + hand-curated 60-key subset .... 75.24   (-0.86)

A high-capacity booster spends its splits chasing noise in near-duplicate
columns; a 3-leaf booster can only afford a column that pays.  This is why the
module ships the FULL evidence vector rather than the curated subset: the
curated subset (``CURATED``, kept for reference) was selected to fix a problem
that was really a capacity problem, and it measures worse than the full vector
at every capacity tested.

WHAT DID NOT WORK, MEASURED AND DISCARDED
-----------------------------------------
*   Greedy forward selection over the evidence families (dm / fa / spl /
    st_core / st_rest / xs_core / xs_field / xs_pair): every single family
    measured negative at 15 leaves, by -0.24 to -0.68.  All of that was the
    capacity artefact above.
*   Seed-bagging the predicted probabilities (5 models, averaged) to reduce
    Brier variance: **exactly zero** change, to the last decimal.
    ``HistGradientBoostingClassifier`` is deterministic given the data at this
    scale - ``random_state`` only perturbs the binning subsample, which is used
    only above 10k rows.  Worth recording so nobody tries it again.
*   Probability calibration (sigmoid or isotonic) on top of the low-capacity
    model: -1.2 to -2.1.  Design D needed a calibrator because its model was
    overconfident; a 3-leaf booster is already well calibrated and wrapping it
    in ``CalibratedClassifierCV`` only throws away 20% of the training data per
    internal fold.
*   ``fa_contradiction`` (a self-contradictory fee receipt) never fires on TRAIN.
    It is retained because it is a policy check that costs nothing and may fire
    on the private set, but it contributes nothing measurable here.

HONEST LIMITS
-------------
The cross-document conflict detectors are high-recall / low-precision against
the *true* risk flags: ``xs_conflict_applicant_name`` catches 76% of true
``identity_conflict`` cases but fires on 189 TRAIN packets to find 26 of them
(precision 0.14).  That is still far better than the stock ``name_agreement``
bit it replaces (precision 0.05, recall 0.59), and precision matters less for a
tree feature than for a rule - but it means these are *hints*, not detections.

The deeper limit is unchanged and is not a modelling problem: 403 of 800 TRAIN
packets contain no biometric slip at all, and their label distribution
(A .33 / D .39 / NR .28) is statistically indistinguishable from the base rate.
The extractor's flag detector is 99.6% precise but only 60% recall, and most of
the missing 40% is simply not printed anywhere in visible OCR.  No feature
engineering recovers evidence the packet does not contain.

COMPLIANCE
----------
*   No case_id is ever keyed on.  The only literal identifiers in this file are
    the three revoked sponsor ids printed in the public FIELD_MANUAL (inherited
    from clf_d) and closed-vocabulary policy enums.
*   Model, calibrator, feature list and every threshold are fit on TRAIN_IDS.
    DEV_IDS is read once, at report time.
*   Only ``common.doc_lines`` (visible OCR) is read.  ``hidden_text`` enters the
    vector solely through extract_v2's disagreement *counts*, which measure
    tampering and never supply a value.
*   Graceful degradation: every new signal is additive and defaults to zero, so
    an unseen private layout falls back to design D's behaviour.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common          # noqa: E402
import extract_v2      # noqa: E402
import clf_d           # noqa: E402
from clf_d import (    # noqa: E402
    CLASSES, Adjudicator, challenge_score, decide, payoff, _sq,
    FA_PENALTY_GRID, _CONFIGS,
)

MODEL_PATH = _HERE / "clf_r2_3_model.joblib"

_FIELDS = ("applicant_name", "species_code", "home_world", "visa_class",
           "sponsor_id", "arrival_date", "declared_purpose", "fee_status")

# page types whose disagreement is policy-meaningful (FIELD_MANUAL precedence)
_SRC_TYPES = ("intake", "registry", "slip", "sponsor", "fee", "note", "correction")


# ---------------------------------------------------------------------------
# F1. cross-source agreement matrix
# ---------------------------------------------------------------------------

def _norm_val(field, val):
    """Normalised, non-redacted reading, or None if the value is unusable."""
    if extract_v2._is_redacted(val):
        return None
    norm = extract_v2.NORMALIZERS.get(field)
    if norm:
        return norm(val)
    v = extract_v2._clean(val)
    if len(extract_v2._alnum(v)) < 2 or extract_v2._looks_like_label(v):
        return None
    return v


def _same(field, a, b):
    if a is None or b is None:
        return None
    if extract_v2.NORMALIZERS.get(field):
        return a == b
    return extract_v2._ratio(extract_v2._alnum(a), extract_v2._alnum(b)) >= 0.80


def cross_source_features(rec):
    """Per-field, per-source agreement structure (F1).

    Uses the extractor's own candidate pool, which already tags every reading
    with the page type that produced it, instead of the single resolved value.
    """
    try:
        cands, page_types = extract_v2._collect_candidates(rec)
    except Exception:
        cands, page_types = {}, []

    out = {}
    n_conf = n_red = n_singleton = 0
    src_conf_counts = {t: 0 for t in _SRC_TYPES}

    for field in _FIELDS:
        entries = cands.get(field, [])
        redacted = 0
        # best reading per (page type, page number): one vote per document
        by_src = {}
        for rank, r, conf, val, ptype, pg in entries:
            if extract_v2._is_redacted(val):
                redacted += 1
                continue
            v = _norm_val(field, val)
            if v is None:
                continue
            score = r * (conf ** 3)
            key = (ptype, pg)
            if key not in by_src or score > by_src[key][0]:
                by_src[key] = (score, v, ptype)

        votes = list(by_src.values())
        # cluster the independent votes into distinct values
        clusters = []          # [total_score, representative, {ptypes}]
        for score, v, ptype in sorted(votes, key=lambda z: -z[0]):
            for c in clusters:
                if _same(field, v, c[1]):
                    c[0] += score
                    c[2].add(ptype)
                    break
            else:
                clusters.append([score, v, {ptype}])
        tot = sum(c[0] for c in clusters) or 1.0
        clusters.sort(key=lambda c: -c[0])

        n_src = len({p for _s, _v, p in votes})
        n_dist = len(clusters)
        top = clusters[0][0] / tot if clusters else 0.0
        # a real conflict: >=2 distinct values, each backed by a different page
        conflict = int(n_dist >= 2 and len(clusters[0][2] | clusters[1][2]) >= 2)

        out[f"xs_nsrc_{field}"] = n_src
        out[f"xs_ndist_{field}"] = n_dist
        out[f"xs_top_{field}"] = float(top)
        out[f"xs_conflict_{field}"] = conflict
        out[f"xs_red_{field}"] = int(redacted > 0)
        out[f"xs_single_{field}"] = int(n_src == 1)
        out[f"xs_absent_{field}"] = int(n_src == 0)

        n_conf += conflict
        n_red += int(redacted > 0)
        n_singleton += int(n_src <= 1)
        if conflict:
            for p in clusters[0][2] | clusters[1][2]:
                if p in src_conf_counts:
                    src_conf_counts[p] += 1

        # explicit pairwise verdicts between the precedence tiers
        best_by_type = {}
        for score, v, ptype in sorted(votes, key=lambda z: -z[0]):
            best_by_type.setdefault(ptype, v)
        for a, b in (("intake", "registry"), ("intake", "slip"),
                     ("intake", "sponsor"), ("registry", "slip")):
            s = _same(field, best_by_type.get(a), best_by_type.get(b))
            out[f"xs_{a}_{b}_{field}"] = 0 if s is None else (1 if s else -1)

    out["xs_n_conflicts"] = n_conf
    out["xs_any_conflict"] = int(n_conf > 0)
    out["xs_n_redacted"] = n_red
    out["xs_n_singleton"] = n_singleton
    for t in _SRC_TYPES:
        out[f"xs_confsrc_{t}"] = src_conf_counts[t]
    # identity-shaped conflict: name/species/home world disagree across pages
    out["xs_identity_conflict"] = int(
        out["xs_conflict_applicant_name"] or out["xs_conflict_species_code"]
        or out["xs_conflict_home_world"])
    out["xs_sponsor_conflict"] = out["xs_conflict_sponsor_id"]
    out["xs_policy_conflict"] = int(out["xs_conflict_visa_class"]
                                    or out["xs_conflict_fee_status"])
    return out, page_types


# ---------------------------------------------------------------------------
# F2. completeness / damage
# ---------------------------------------------------------------------------

# The label inventory each page type prints when it is undamaged.  Closed
# vocabulary from the forms themselves, not applicant data.
_PAGE_LABELS = {
    "intake": ("caseid", "applicant", "speciescode", "homeworld", "visaclass",
               "sponsorid", "arrivaldate", "declaredpurpose"),
    "registry": ("registryname", "homeworld", "speciescode", "registrystatus",
                 "arrivaldate"),
    "slip": ("caseid", "applicant", "speciesmatch", "biometricconfidence",
             "observedflags"),
    "fee": ("caseid", "feestatus", "amount", "waivercode"),
    "note": ("finding", "reason"),
}

_DAMAGE_WORDS = ("washedout", "whiteout", "redacted", "illegible", "torn",
                 "unreadable", "missing", "damaged", "obscured")
_BRACKET_RE = re.compile(r"\[[^\]]{2,40}\]")


def damage_features(rec, page_types):
    """Structural completeness of the packet (F2)."""
    pages = common.pages_of(rec)
    n_pages = max(len(pages), 1)
    own = extract_v2._id_digits(str(rec.get("case_id", "")).split("-")[-1])

    n_blank = n_foreign = 0
    cov_sum = cov_n = 0.0
    missing_lbl = 0
    lowconf = tot_lines = 0
    conf_vals = []
    type_counts = {}
    intake_cov = registry_cov = slip_cov = fee_cov = -1.0

    for p, ptype in zip(pages, list(page_types) + ["other"] * len(pages)):
        lines = common.page_lines(p)
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        body = [l for l in lines if "synthetichiring" not in _sq(l["t"])
                and not _sq(l["t"]).startswith("packetmib")]
        if len(body) <= 1:
            n_blank += 1
        try:
            if lines and extract_v2.page_is_foreign([dict(l) for l in lines], own):
                n_foreign += 1
        except Exception:
            pass
        joined = _sq(" ".join(l["t"] for l in lines))
        want = _PAGE_LABELS.get(ptype)
        if want:
            hit = sum(1 for w in want if w in joined)
            cov = hit / len(want)
            cov_sum += cov
            cov_n += 1
            missing_lbl += len(want) - hit
            if ptype == "intake":
                intake_cov = max(intake_cov, cov)
            elif ptype == "registry":
                registry_cov = max(registry_cov, cov)
            elif ptype == "slip":
                slip_cov = max(slip_cov, cov)
            elif ptype == "fee":
                fee_cov = max(fee_cov, cov)
        for l in lines:
            tot_lines += 1
            conf_vals.append(float(l.get("s", 1.0)))
            if float(l.get("s", 1.0)) < 0.80:
                lowconf += 1

    vis = " ".join(l["t"] for l in common.doc_lines(rec))
    vis_sq = _sq(vis)
    brackets = _BRACKET_RE.findall(vis)
    dmg_words = sum(1 for w in _DAMAGE_WORDS if w in vis_sq)

    out = {
        "dm_n_blank_pages": n_blank,
        "dm_blank_frac": n_blank / n_pages,
        "dm_n_foreign_pages": n_foreign,
        "dm_any_foreign": int(n_foreign > 0),
        "dm_label_coverage": (cov_sum / cov_n) if cov_n else 0.0,
        "dm_typed_pages": cov_n,
        "dm_untyped_pages": type_counts.get("other", 0),
        "dm_missing_labels": missing_lbl,
        "dm_intake_cov": intake_cov,
        "dm_registry_cov": registry_cov,
        "dm_slip_cov": slip_cov,
        "dm_fee_cov": fee_cov,
        "dm_has_intake": int("intake" in type_counts),
        "dm_has_registry": int("registry" in type_counts),
        "dm_has_slip": int("slip" in type_counts),
        "dm_has_sponsor": int("sponsor" in type_counts),
        "dm_has_fee": int("fee" in type_counts),
        "dm_n_note_pages": type_counts.get("note", 0),
        "dm_dup_intake": int(type_counts.get("intake", 0) > 1),
        "dm_dup_note": int(type_counts.get("note", 0) > 1),
        "dm_n_brackets": len(brackets),
        "dm_any_bracket": int(bool(brackets)),
        "dm_damage_words": dmg_words,
        "dm_lowconf_lines": lowconf,
        "dm_lowconf_frac": lowconf / max(tot_lines, 1),
        "dm_conf_std": float(np.std(conf_vals)) if conf_vals else 0.0,
        "dm_conf_p25": float(np.percentile(conf_vals, 25)) if conf_vals else 1.0,
        "dm_lines_per_page": tot_lines / n_pages,
    }
    # how many of the five evidence documents the packet actually contains
    out["dm_n_doc_types"] = sum(out[f"dm_has_{t}"] for t in
                                ("intake", "registry", "slip", "sponsor", "fee"))
    return out


# ---------------------------------------------------------------------------
# F3. stamp geometry vs watermark
# ---------------------------------------------------------------------------

# short annotation tokens whose *geometry* carries the meaning
_STAMP_TOKENS = {
    "denied": ("denied", "denial"),
    "approved": ("approved", "approval"),
    "needsreview": ("needsreview",),
    "sampledenial": ("sampledenial",),
    "rescind": ("rescinded", "rescind"),
    "void": ("void", "revoked"),
    "casework": ("casework",),
    "archive": ("archive", "archivecopy"),
    "redacted": ("redacted",),
    "copyartifact": ("copyartifact", "duplicate"),
    "embargo": ("embargo",),
    "hardship": ("hardship",),
}

_MAX_STAMP_LEN = 34          # stamps and watermarks are short, standalone lines


def stamp_features(rec, page_types):
    """Geometry of every decision-ish token: stamp or watermark? (F3)"""
    pages = common.pages_of(rec)
    n_pages = max(len(pages), 1)
    note_pages = {i for i, t in enumerate(page_types) if t == "note"}

    acc = {k: {"n": 0, "pages": set(), "maxh": 0.0, "conf": 0.0,
               "on_note": 0, "alone": 0, "maxw": 0.0}
           for k in _STAMP_TOKENS}

    for l in common.doc_lines(rec):
        s = _sq(l["t"])
        if not s or len(s) > _MAX_STAMP_LEN:
            continue
        ph = float(l.get("ph") or 1.0) or 1.0
        pw = float(l.get("pw") or 1.0) or 1.0
        h = (float(l["y1"]) - float(l["y0"])) / ph
        w = (float(l["x1"]) - float(l["x0"])) / pw
        conf = float(l.get("s", 1.0))
        pg = int(l.get("page", 0))
        for key, pats in _STAMP_TOKENS.items():
            hit = False
            for pat in pats:
                if pat in s:
                    hit = True
                    break
                # OCR-shredded stamps ("SAMPLFDFNIA]", "DENlED"): fuzzy match
                if abs(len(s) - len(pat)) <= max(3, len(pat) // 3):
                    if extract_v2._ratio(s, pat) >= 0.78:
                        hit = True
                        break
            if not hit:
                continue
            a = acc[key]
            a["n"] += 1
            a["pages"].add(pg)
            a["maxh"] = max(a["maxh"], h)
            a["maxw"] = max(a["maxw"], w)
            a["conf"] = max(a["conf"], conf)
            a["on_note"] |= int(pg in note_pages)
            a["alone"] |= int(len(s) <= len(pats[0]) + 4)

    out = {}
    for key, a in acc.items():
        npg = len(a["pages"])
        out[f"st_{key}_n"] = a["n"]
        out[f"st_{key}_pages"] = npg
        out[f"st_{key}_pagefrac"] = npg / n_pages
        out[f"st_{key}_maxh"] = a["maxh"]
        out[f"st_{key}_maxw"] = a["maxw"]
        out[f"st_{key}_conf"] = a["conf"]
        out[f"st_{key}_on_note"] = a["on_note"]
        out[f"st_{key}_alone"] = a["alone"]
        # a watermark repeats across most of the packet; a stamp does not
        out[f"st_{key}_watermarky"] = int(npg >= 2 and npg / n_pages >= 0.5)
        out[f"st_{key}_stampy"] = int(npg == 1 and a["n"] <= 2 and a["alone"])

    # the FIELD_MANUAL "sample denial watermark" trap, geometrically identified
    out["st_denial_wm_trap"] = int(out["st_sampledenial_pages"] >= 1
                                   and out["st_sampledenial_pagefrac"] >= 0.4)
    # a denial word that is NOT explained by the sample-denial watermark
    out["st_real_denial_mark"] = int(out["st_denied_n"] > 0
                                     and out["st_sampledenial_n"] == 0)
    # both an approval and a denial mark on the packet: trap #2 (crossed-out stamp)
    out["st_appr_and_denial"] = int(out["st_denied_n"] > 0
                                    and out["st_approved_n"] > 0)
    return out


# ---------------------------------------------------------------------------
# F4. sponsor-attestation semantic check
# ---------------------------------------------------------------------------

_SPN_RE = re.compile(r"SPN[-_ ]?(\d{4})", re.I)
_CLASS_RE = re.compile(r"\b(XW-?[12]|DIP-?1|MED-?3|TRANSIT-?7)\b", re.I)
_ATTEST_RE = re.compile(r"attests?\s+that\s+(.{2,60}?)\s+is\s+expected", re.I)


def sponsor_letter_features(rec, page_types, fields):
    """Parse the attestation prose and cross-check it (F4)."""
    out = {"spl_present": 0, "spl_spn_found": 0, "spl_spn_match": 0,
           "spl_name_found": 0, "spl_name_match": 0, "spl_class_found": 0,
           "spl_class_match": 0, "spl_caseid_clause": 0, "spl_n_mismatch": 0,
           "spl_manual_revoked": 0}
    pages = common.pages_of(rec)
    text_parts = []
    for p, t in zip(pages, list(page_types) + ["other"] * len(pages)):
        if t == "sponsor":
            text_parts.append(" ".join(l["t"] for l in common.page_lines(p)))
    if not text_parts:
        return out
    txt = " ".join(text_parts)
    out["spl_present"] = 1

    mism = 0
    m = _SPN_RE.search(txt)
    if m:
        out["spl_spn_found"] = 1
        spn = "SPN-" + m.group(1)
        out["spl_manual_revoked"] = int(spn in clf_d.MANUAL_REVOKED_SPONSORS)
        want = str(fields.get("sponsor_id", ""))
        if re.fullmatch(r"SPN-\d{4}", want):
            ok = int(spn == want)
            out["spl_spn_match"] = ok
            mism += 1 - ok

    m = _ATTEST_RE.search(txt)
    if m:
        out["spl_name_found"] = 1
        got = extract_v2._alnum(m.group(1))
        want = extract_v2._alnum(str(fields.get("applicant_name", "")))
        if len(want) > 3 and len(got) > 3:
            ok = int(extract_v2._ratio(got, want) >= 0.80)
            out["spl_name_match"] = ok
            mism += 1 - ok

    cls = None
    mc = re.search(r"class\s+([A-Za-z0-9\-]{3,10})", txt, re.I)
    if mc:
        c2 = _CLASS_RE.search(mc.group(1))
        cls = c2.group(1).upper() if c2 else None
    if cls is None:
        c2 = _CLASS_RE.search(txt)
        cls = c2.group(1).upper() if c2 else None
    if cls:
        cls = cls.replace("XW", "XW-").replace("DIP", "DIP-").replace(
            "MED", "MED-").replace("TRANSIT", "TRANSIT-").replace("--", "-")
        out["spl_class_found"] = 1
        want = str(fields.get("visa_class", ""))
        if want and want != "unknown":
            ok = int(cls == want)
            out["spl_class_match"] = ok
            mism += 1 - ok

    out["spl_caseid_clause"] = int("activecaseid" in _sq(txt))
    out["spl_n_mismatch"] = mism
    return out


# ---------------------------------------------------------------------------
# F5. fee-receipt arithmetic
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"\$\s?([0-9][0-9,]*\.?\d{0,2})")


def fee_arith_features(rec, page_types, fields):
    """Is the fee receipt internally consistent? (F5)"""
    out = {"fa_amount_found": 0, "fa_amount_zero": 0, "fa_amount_pos": 0,
           "fa_amount_log": 0.0, "fa_waivercode_na": 0, "fa_waivercode_real": 0,
           "fa_paid_zero": 0, "fa_waived_pos": 0, "fa_unpaid_pos": 0,
           "fa_receipt_present": 0, "fa_contradiction": 0}
    pages = common.pages_of(rec)
    parts = [" ".join(l["t"] for l in common.page_lines(p))
             for p, t in zip(pages, list(page_types) + ["other"] * len(pages))
             if t == "fee"]
    if not parts:
        return out
    txt = " ".join(parts)
    out["fa_receipt_present"] = 1
    sq = _sq(txt)

    m = _AMOUNT_RE.search(txt)
    amt = None
    if m:
        try:
            amt = float(m.group(1).replace(",", ""))
        except ValueError:
            amt = None
    if amt is not None:
        out["fa_amount_found"] = 1
        out["fa_amount_zero"] = int(amt == 0.0)
        out["fa_amount_pos"] = int(amt > 0.0)
        out["fa_amount_log"] = float(np.log1p(amt))
    elif "amountna" in sq:
        out["fa_amount_found"] = 1

    out["fa_waivercode_na"] = int("waivercodena" in sq)
    mw = re.search(r"waivercode([a-z0-9]{2,12})", sq)
    if mw and mw.group(1) not in ("na", "none"):
        out["fa_waivercode_real"] = 1

    fee = str(fields.get("fee_status", ""))
    if amt is not None:
        out["fa_paid_zero"] = int(fee == "paid" and amt == 0.0)
        out["fa_waived_pos"] = int(fee == "waived" and amt > 0.0
                                   and not out["fa_waivercode_real"])
        out["fa_unpaid_pos"] = int(fee == "unpaid" and amt > 0.0)
    out["fa_contradiction"] = int(out["fa_paid_zero"] or out["fa_waived_pos"])
    return out


# ---------------------------------------------------------------------------
# full feature vector
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# F6. biometric scan quality
# ---------------------------------------------------------------------------

_BIOCONF_RE = re.compile(r"onfidence\s*[:\s]\s*([0-9]{1,3})\s*%")


def biometric_features(rec):
    """The slip prints ``Biometric confidence: NN%``.

    Measured on TRAIN: below 80% the packet carries ``illegible_biometrics``
    42% of the time, at or above 90% only 5% of the time.  The stock vector
    never reads this number, and it is the one piece of *graded* evidence in the
    packet - everything else is categorical.
    """
    out = {"bio_conf_found": 0, "bio_conf": 0.0, "bio_conf_lt80": 0,
           "bio_conf_lt90": 0, "bio_conf_ge95": 0}
    m = _BIOCONF_RE.search(common.visible_text(rec))
    if not m:
        return out
    try:
        v = float(m.group(1))
    except ValueError:
        return out
    if not 0 <= v <= 100:
        return out
    out["bio_conf_found"] = 1
    out["bio_conf"] = v
    out["bio_conf_lt80"] = int(v < 80)
    out["bio_conf_lt90"] = int(v < 90)
    out["bio_conf_ge95"] = int(v >= 95)
    return out


def evidence_features(rec, fields):
    """The new evidence families (F1-F6). Visible OCR only."""
    xs, page_types = cross_source_features(rec)
    f = dict(xs)
    f.update(damage_features(rec, page_types))
    f.update(stamp_features(rec, page_types))
    f.update(sponsor_letter_features(rec, page_types, fields))
    f.update(fee_arith_features(rec, page_types, fields))
    f.update(biometric_features(rec))
    return f


# The subset of the evidence vector that survived TRAIN-only selection.  The
# full 280-key dump measurably *hurts* (see __main__): with 800 training rows a
# gradient booster spends its splits on near-duplicate columns.  These are the
# keys with standalone univariate lift plus the interactions.
CURATED = (
    # F3 stamp geometry - the strongest new family
    "st_real_denial_mark", "st_denial_wm_trap", "st_appr_and_denial",
    "st_denied_n", "st_denied_pages", "st_denied_pagefrac", "st_denied_maxh",
    "st_denied_on_note", "st_denied_stampy", "st_denied_watermarky",
    "st_approved_n", "st_approved_pages", "st_approved_on_note",
    "st_sampledenial_n", "st_sampledenial_pagefrac",
    "st_rescind_n", "st_rescind_pages", "st_void_n", "st_embargo_n",
    "st_needsreview_n",
    # F2 completeness / damage
    "dm_n_doc_types", "dm_label_coverage", "dm_missing_labels",
    "dm_n_blank_pages", "dm_any_bracket", "dm_n_brackets", "dm_damage_words",
    "dm_lowconf_frac", "dm_conf_p25", "dm_any_foreign", "dm_n_note_pages",
    "dm_has_slip", "dm_has_sponsor", "dm_has_registry", "dm_intake_cov",
    # F1 cross-source agreement (aggregates + the two identity-shaped bits)
    "xs_n_conflicts", "xs_any_conflict", "xs_identity_conflict",
    "xs_sponsor_conflict", "xs_policy_conflict", "xs_n_redacted",
    "xs_top_applicant_name", "xs_top_sponsor_id", "xs_top_visa_class",
    "xs_top_fee_status", "xs_nsrc_applicant_name",
    # F4 / F5 / F6
    "spl_present", "spl_n_mismatch", "spl_spn_match", "spl_name_match",
    "spl_class_match", "spl_manual_revoked",
    "fa_amount_zero", "fa_amount_log", "fa_waivercode_real", "fa_paid_zero",
    "bio_conf_found", "bio_conf", "bio_conf_lt80", "bio_conf_ge95",
)


def r2_features(rec, fields, base_feats, design=None):
    """design D vector (97 stock + note/policy) plus the new evidence families."""
    f = dict(design) if design is not None else clf_d.design_features(
        rec, fields, base_feats)
    f.update(evidence_features(rec, fields))
    # a couple of interactions the trees would need many splits to find
    f["ix_conflict_and_no_note"] = int(f.get("xs_any_conflict", 0)
                                       and not f.get("note_find_any", 0))
    f["ix_clean_and_complete"] = int(
        f.get("slip_clean", 0) and f.get("dm_n_doc_types", 0) >= 4
        and not f.get("xs_any_conflict", 0) and not f.get("dm_any_bracket", 0))
    f["ix_damaged_evidence"] = int(
        f.get("dm_any_bracket", 0) or f.get("dm_n_blank_pages", 0)
        or f.get("dm_lowconf_frac", 0.0) > 0.25)
    return f


# ---------------------------------------------------------------------------
# model head: same family as design D, at a capacity chosen on TRAIN
# ---------------------------------------------------------------------------
#
# The single biggest measured effect in this round was NOT a feature.  Design D
# selected ``max_leaf_nodes=15`` on a single 5-fold split; averaged over five
# paired fold seeds that turns out to be well past the overfitting point for 800
# rows.  Dropping to 3-4 leaves is worth ~+1.8 TRAIN OOF points on the *same*
# feature vector, and it also flips the verdict on the evidence features: at 15
# leaves the extra columns cost ~1 point, at 3 leaves they gain ~0.6, because a
# high-capacity booster spends its splits chasing noise in near-duplicate
# columns while a stump-depth booster can only use a column that pays.
#
# Candidates below are scored by mean TRAIN out-of-fold score over five paired
# fold seeds.  DEV is never involved.
_R2_CONFIGS = [
    {"calib": None, "max_iter": 500, "lr": 0.03, "leaves": 3, "msl": 20, "l2": 1.0},
    {"calib": None, "max_iter": 300, "lr": 0.05, "leaves": 3, "msl": 20, "l2": 1.0},
    {"calib": None, "max_iter": 300, "lr": 0.05, "leaves": 4, "msl": 20, "l2": 1.0},
    {"calib": None, "max_iter": 300, "lr": 0.05, "leaves": 2, "msl": 20, "l2": 1.0},
    {"calib": None, "max_iter": 300, "lr": 0.05, "leaves": 4, "msl": 20,
     "l2": 1.0, "mf": 0.7},
]
_SEEDS = (0, 1, 2, 3, 4)


class R2Adjudicator(Adjudicator):
    """design D's Adjudicator with min_samples_leaf / max_features exposed."""

    def _make(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        c = self.config
        base = HistGradientBoostingClassifier(
            max_iter=c["max_iter"], learning_rate=c["lr"],
            max_leaf_nodes=c["leaves"], min_samples_leaf=c.get("msl", 20),
            max_features=c.get("mf", 1.0),
            l2_regularization=c.get("l2", 1.0), random_state=self.seed)
        if c.get("calib"):
            from sklearn.calibration import CalibratedClassifierCV
            return CalibratedClassifierCV(base, method=c["calib"], cv=5)
        return base

    def predict_one(self, rec, fields, base_feats):
        """End-to-end entry point for the production pipeline.

        Must override design D's, which builds only the design-D vector; this
        model is fitted on the design-D vector PLUS the evidence families.
        """
        lab, conf = self.predict([r2_features(rec, fields, base_feats)])
        return lab[0], float(conf[0])

    @staticmethod
    def oof_proba(feats_list, labels, config, seed=0, folds=5):
        from sklearn.model_selection import StratifiedKFold
        keys = sorted({k for d in feats_list for k in d})
        y = np.array([CLASSES.index(l) for l in labels])
        out = np.zeros((len(feats_list), len(CLASSES)))
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for a, b in skf.split(np.zeros(len(y)), y):
            m = R2Adjudicator(seed=seed, config=config)
            m.fit([feats_list[i] for i in a], [labels[i] for i in a], keys=keys)
            out[b] = m.predict_proba([feats_list[i] for i in b])
        return out


# ---------------------------------------------------------------------------
# self-test / report
# ---------------------------------------------------------------------------

def _cache_pickle(name):
    return _HERE.parent / name


def _load_featcache():
    import pickle
    pkl = _cache_pickle("featcache_d.pkl")
    if pkl.exists():
        return pickle.load(open(pkl, "rb"))
    cache = common.load_cache()
    out = {}
    for cid, rec in sorted(cache.items()):
        fl = extract_v2.extract_fields_v2(rec)
        out[cid] = (fl, extract_v2.extract_features(rec, fl))
    pickle.dump(out, open(pkl, "wb"))
    return out


def _design_cache(cache, fc, ids):
    import pickle
    pkl = _cache_pickle("designfeat_d.pkl")
    have = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    todo = [c for c in ids if c not in have]
    for n, c in enumerate(todo):
        have[c] = clf_d.design_features(cache[c], fc[c][0], fc[c][1])
        if n and n % 200 == 0:
            print(f"    ... design features {n}/{len(todo)}", flush=True)
    if todo:
        pickle.dump(have, open(pkl, "wb"))
    return have


def _evidence_cache(cache, fc, ids):
    import pickle
    pkl = _cache_pickle("evidfeat_r2_3.pkl")
    have = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    todo = [c for c in ids if c not in have]
    for n, c in enumerate(todo):
        have[c] = evidence_features(cache[c], fc[c][0])
        if n and n % 200 == 0:
            print(f"    ... evidence features {n}/{len(todo)}", flush=True)
    if todo:
        pickle.dump(have, open(pkl, "wb"))
    return have


def _combine(design, evid):
    f = dict(design)
    f.update(evid)
    f["ix_conflict_and_no_note"] = int(f.get("xs_any_conflict", 0)
                                       and not f.get("note_find_any", 0))
    f["ix_clean_and_complete"] = int(
        f.get("slip_clean", 0) and f.get("dm_n_doc_types", 0) >= 4
        and not f.get("xs_any_conflict", 0) and not f.get("dm_any_bracket", 0))
    f["ix_damaged_evidence"] = int(
        f.get("dm_any_bracket", 0) or f.get("dm_n_blank_pages", 0)
        or f.get("dm_lowconf_frac", 0.0) > 0.25)
    return f


_RULES = ("argmax",) + tuple(f"EV{int(-p)}" for p in FA_PENALTY_GRID)


def _score_rules(pr, Y):
    am = [CLASSES[i] for i in pr.argmax(1)]
    s = challenge_score(Y, am, pr.max(1))
    out = {"argmax": (s["classification"] + s["calibration"],
                      s["false_approvals"])}
    for p in FA_PENALTY_GRID:
        lab, cf = decide(pr, p)
        s2 = challenge_score(Y, lab, cf)
        out[f"EV{int(-p)}"] = (s2["classification"] + s2["calibration"],
                               s2["false_approvals"])
    return out


def _select(Ftr, Ytr, cfgs=_R2_CONFIGS, seeds=_SEEDS, tag=""):
    """Model + decision-rule selection on TRAIN out-of-fold only.

    Scores are averaged over `seeds` *paired* fold splits.  A single 5-fold
    split has a per-seed standard deviation of roughly 0.5 points on this
    metric, which is the same size as the effects being compared - design D
    selected its configuration on one split and landed on an over-capacity
    model as a result.
    """
    best = None
    for cfg in cfgs:
        per = [_score_rules(R2Adjudicator.oof_proba(Ftr, Ytr, cfg, seed=s), Ytr)
               for s in seeds]
        line = (f"    {tag}lv={cfg['leaves']} it={cfg['max_iter']:3d} "
                f"lr={cfg['lr']:.2f} mf={cfg.get('mf', 1.0):.1f} |")
        for r in _RULES:
            m = float(np.mean([p[r][0] for p in per]))
            fa = float(np.mean([p[r][1] for p in per]))
            line += f" {r} {m:6.2f}(fa {fa:4.1f})"
            if best is None or m > best[0]:
                pen = -4.0 if r == "argmax" else -float(r[2:])
                best = (m, cfg, pen, "argmax" if r == "argmax" else "ev")
        print(line, flush=True)
    return best


def _main():
    labels = common.load_labels()
    cache = common.load_cache()
    fc = _load_featcache()
    train, dev = common.split_ids(labels)
    train = [c for c in train if c in cache and c in fc]
    dev = [c for c in dev if c in cache and c in fc]
    print(f"cache={len(cache)}  train={len(train)}  dev={len(dev)}")

    print("  building design-D features ...", flush=True)
    D = _design_cache(cache, fc, train + dev)
    print("  building evidence features (F1-F5) ...", flush=True)
    E = _evidence_cache(cache, fc, train + dev)
    F = {c: _combine(D[c], E[c]) for c in train + dev}
    print(f"  feature count: design-D {len(D[train[0]])}  "
          f"+ evidence {len(E[train[0]])}  = {len(F[train[0]])}")

    Ytr = [labels[c]["adjudication"] for c in train]
    Ytr_arr = np.array(Ytr)

    # ---- evidence sanity: do the new conflict bits actually predict? --------
    print("\n  new-feature sanity on TRAIN (rate of each class when the bit fires):")
    for k in ("xs_identity_conflict", "xs_sponsor_conflict", "xs_any_conflict",
              "dm_any_bracket", "dm_n_blank_pages", "st_denial_wm_trap",
              "st_real_denial_mark", "spl_n_mismatch", "fa_contradiction",
              "ix_clean_and_complete"):
        v = np.array([float(F[c].get(k, 0.0)) for c in train])
        on = v > 0
        if on.sum() == 0:
            print(f"    {k:24s} never fires")
            continue
        rates = {cl: float((Ytr_arr[on] == cl).mean()) for cl in CLASSES}
        base = {cl: float((Ytr_arr == cl).mean()) for cl in CLASSES}
        print(f"    {k:24s} n={int(on.sum()):4d} "
              f"A {rates['APPROVED']:.2f} D {rates['DENIED']:.2f} "
              f"NR {rates['NEEDS_REVIEW']:.2f}   (base A {base['APPROVED']:.2f} "
              f"D {base['DENIED']:.2f} NR {base['NEEDS_REVIEW']:.2f})")

    # ---- ablation, TRAIN out-of-fold only ----------------------------------
    print("\n  TRAIN out-of-fold ablation, mean over 5 paired fold seeds "
          "(dev never touched):")
    print("   [0] design-D features AT DESIGN-D's OWN CAPACITY (lv=15) "
          "- the incumbent")
    best_0 = _select([D[c] for c in train], Ytr,
                     cfgs=[{"calib": None, "max_iter": 250, "lr": 0.05,
                            "leaves": 15, "msl": 20, "l2": 1.0}])
    print(f"   -> incumbent TRAIN OOF cls+cal {best_0[0]:.2f}")
    print("   [A] design-D features, low-capacity grid")
    best_d = _select([D[c] for c in train], Ytr)
    print(f"   -> design-D best TRAIN OOF cls+cal {best_d[0]:.2f} "
          f"(capacity alone: {best_d[0]-best_0[0]:+.2f})")
    print("   [B] design-D + evidence features, low-capacity grid")
    best_f = _select([F[c] for c in train], Ytr)
    print(f"   -> combined best TRAIN OOF cls+cal {best_f[0]:.2f} "
          f"(evidence features: {best_f[0]-best_d[0]:+.2f}; "
          f"total vs incumbent {best_f[0]-best_0[0]:+.2f})")

    _tot, cfg, pen, rule = best_f
    print(f"\n  selected {cfg}, rule={rule}, fa_penalty={pen}")

    clf = R2Adjudicator(config=cfg, fa_penalty=pen, rule=rule).fit(
        [F[c] for c in train], Ytr)
    print(f"  fitted on {len(train)} TRAIN cases, {len(clf.keys)} features")

    # ---- DEV report (read once) -------------------------------------------
    proba = clf.predict_proba([F[c] for c in dev])
    argmax = [CLASSES[i] for i in proba.argmax(1)]
    if rule == "argmax":
        adj, conf = argmax, np.clip(proba.max(1), 0.01, 0.99)
    else:
        adj, conf = decide(proba, pen)

    truth = [labels[c]["adjudication"] for c in dev]
    print(f"\n  dev accuracy: argmax "
          f"{np.mean(np.array(argmax)==np.array(truth)):.3f}"
          f"   EV-rule {np.mean(np.array(adj)==np.array(truth)):.3f}")
    cm = {}
    for t, p in zip(truth, adj):
        cm[(t, p)] = cm.get((t, p), 0) + 1
    print("  dev confusion (truth -> pred):")
    for t in CLASSES:
        print("   ", t.ljust(13),
              "  ".join(f"{p[:4]}={cm.get((t,p),0):3d}" for p in CLASSES))

    rows = []
    for c, a, cf in zip(dev, adj, conf):
        r = {"case_id": c, **{k: v for k, v in fc[c][0].items() if k != "_meta"}}
        r["adjudication"] = a
        r["confidence"] = round(float(cf), 4)
        rows.append(r)
    res = common.score_rows(rows, "clf_r2_3_dev")
    print("\n  DEV  " + common.summarize(res))

    clf.save(MODEL_PATH)
    print(f"  saved model -> {MODEL_PATH} "
          f"({MODEL_PATH.stat().st_size/1e6:.1f} MB)")
    return res


if __name__ == "__main__":
    _main()
