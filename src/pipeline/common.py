"""Shared foundation for the MIB pipeline upgrade. ALL agents import this.

Owns the ONE deterministic train/dev split so every experiment is comparable and
no one accidentally measures on data they fit on.

Rules that must never be broken (see EVALUATION.md):
  * Nothing may be keyed to a specific case_id. Learn general patterns only.
  * `hidden_text` (PDF text layer) is the injection surface. It is the LOWEST
    precedence evidence (FIELD_MANUAL rank 6) and must NEVER supply a value that
    visible OCR does not corroborate.
  * Fit anything (vocab, models, thresholds) on TRAIN_IDS only; report on DEV_IDS.
"""
import csv, json, os, subprocess, sys
from pathlib import Path

_DEV_HOST_REPO = Path("/Users/weitu/Desktop/BUILD/mib-doc-challenge")


def _find_repo():
    """Locate the repo WITHOUT requiring it to exist.

    Order: $MIB_REPO -> a parent of this module that looks like the repo ->
    the original dev-host path as a last resort. Never raises, never touches
    the filesystem in a way that can fail: inference must import cleanly on a
    container where none of these exist.
    """
    env = os.environ.get("MIB_REPO")
    if env:
        return Path(env).expanduser()
    try:
        here = Path(__file__).resolve()
        for d in here.parents:
            if (d / "scripts" / "evaluate.py").exists() or (d / "data" / "train_labels.csv").exists():
                return d
    except OSError:
        pass
    return _DEV_HOST_REPO


REPO = _find_repo()
OUT = Path(os.environ.get("MIB_OUT") or (REPO / "notebooks" / "_out"))
CACHE = Path(os.environ.get("MIB_OCR_CACHE") or (OUT / "ocr_cache.jsonl"))
# Dev-time only. Resolved lazily (see _require_scoring_paths) so that importing
# this module, and running inference, never depends on them existing.
LABELS_CSV = Path(os.environ.get("MIB_LABELS") or (REPO / "data" / "train_labels.csv"))
EVALUATE = Path(os.environ.get("MIB_EVALUATE") or (REPO / "scripts" / "evaluate.py"))


def _python():
    """Interpreter used to shell out to evaluate.py. Falls back to our own."""
    cand = Path(os.environ.get("MIB_PY") or (REPO / ".venv" / "bin" / "python"))
    try:
        if cand.exists():
            return str(cand)
    except OSError:
        pass
    return sys.executable or "python3"

# EVALUATION.md field extraction table
FIELD_WEIGHTS = {
    "applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
    "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3, "risk_flags": 8,
    "fee_status": 4,
}
FIELDNAMES = ["case_id", "applicant_name", "species_code", "home_world", "visa_class",
              "sponsor_id", "arrival_date", "declared_purpose", "risk_flags", "fee_status",
              "adjudication", "confidence"]

DISQUALIFYING = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
REVIEW_ONLY = {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}


def load_labels(path=None):
    """Ground-truth labels. Dev-time only - inference must never call this."""
    p = Path(path or LABELS_CSV)
    with open(p, newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def load_cache(limit=None):
    """case_id -> {case_id, scale, pages:[{page,k,w,h,lines:[...],hidden_text}]}"""
    out = {}
    if not CACHE.exists():
        return out
    with open(CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # tolerate a torn last line while cache is still building
            out[rec["case_id"]] = rec
            if limit and len(out) >= limit:
                break
    return out


_LINE_NUM_KEYS = ("s", "x0", "y0", "x1", "y1")


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default        # NaN -> default


def pages_of(rec):
    """The record's pages, defensively. A malformed record yields no pages."""
    if not isinstance(rec, dict):
        return []
    ps = rec.get("pages")
    if not isinstance(ps, list):
        return []
    return [p for p in ps if isinstance(p, dict)]


def page_lines(p):
    """Lines of ONE page as fresh dicts with every expected key present.

    A partial/damaged page degrades to 'no lines' (or a line with empty text and
    zero geometry) instead of raising: losing a page costs a few fields, whereas
    an exception loses the entire case.
    """
    if not isinstance(p, dict):
        return []
    raw = p.get("lines")
    if not isinstance(raw, list):
        return []
    out = []
    for l in raw:
        if not isinstance(l, dict):
            continue
        d = dict(l)
        t = d.get("t")
        d["t"] = "" if t is None else str(t)
        for k in _LINE_NUM_KEYS:
            d[k] = _num(d.get(k))
        out.append(d)
    return out


def doc_lines(rec):
    """Flat list of visible OCR lines; each gets its page index. TRUSTED evidence."""
    out = []
    for p in pages_of(rec):
        pg = p.get("page", 0)
        pw = _num(p.get("w"), 1.0) or 1.0
        ph = _num(p.get("h"), 1.0) or 1.0
        for d in page_lines(p):
            d["page"] = pg; d["pw"] = pw; d["ph"] = ph
            out.append(d)
    return out


def visible_text(rec):
    """Concatenated visible OCR text. TRUSTED."""
    return " ".join(l["t"] for l in doc_lines(rec))


def hidden_text(rec):
    """PDF text layer. UNTRUSTED / injection surface - never a value source."""
    return " ".join(str(p.get("hidden_text", "") or "") for p in pages_of(rec))


def split_ids(case_ids):
    """Deterministic 80/20 train/dev split by case number. No randomness, no leakage."""
    train, dev = [], []
    for cid in sorted(case_ids):
        n = int(cid.split("-")[1])
        (dev if n % 5 == 0 else train).append(cid)
    return train, dev


def score_rows(rows, tag="run", workdir=None):
    """Score prediction rows with the REAL scripts/evaluate.py.

    Builds a truth CSV of exactly the scored cases so a subset isn't unfairly
    penalized as 'missing'. Returns evaluate.py's parsed results dict.
    """
    labels_csv, evaluate = Path(LABELS_CSV), Path(EVALUATE)
    missing = [str(p) for p in (labels_csv, evaluate) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "score_rows() is a DEV-TIME helper and needs the repo. Missing: "
            + ", ".join(missing) + ". Set $MIB_REPO (or $MIB_LABELS/$MIB_EVALUATE). "
            "Inference does not require these.")

    workdir = Path(workdir or OUT); workdir.mkdir(parents=True, exist_ok=True)
    preds = workdir / f"predictions_{tag}.jsonl"
    truth = workdir / f"truth_{tag}.csv"
    evj = workdir / f"evaluation_{tag}.json"

    with open(preds, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    ids = {r["case_id"] for r in rows}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f); header = reader.fieldnames
        keep = [r for r in reader if r["case_id"] in ids]
    with open(truth, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(keep)

    subprocess.run([_python(), str(evaluate), "--truth", str(truth), "--submission", str(preds),
                    "--output-json", str(evj)], check=False, capture_output=True)
    return json.loads(evj.read_text())


def summarize(res):
    s = res["scores"]
    return (f"TOTAL {s['total_score']:.2f}/150 | class {s['classification_score']:.2f}/80 "
            f"| extract {s['extraction_score']:.2f}/50 | calib {s['calibration_score']:.2f}/20 "
            f"| false_approvals {res['raw']['catastrophic_false_approvals']}")


if __name__ == "__main__":
    cache = load_cache(); labels = load_labels()
    tr, dv = split_ids(labels)
    print(f"cache={len(cache)} labels={len(labels)} train={len(tr)} dev={len(dv)}")
    if cache:
        cid = next(iter(cache))
        print(f"sample {cid}: {len(doc_lines(cache[cid]))} visible lines, "
              f"{len(hidden_text(cache[cid]))} hidden chars")
