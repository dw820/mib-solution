#!/usr/bin/env python3
"""Entry point: read PDFs from an input dir, write predictions to an output path.

Orchestration only. Per PDF:  OCR (src/ocr) -> field extraction (extract_v2)
-> adjudication + confidence (clf_final).  See README.md / MEMO.md for the design.

Work is parallelized across the vCPUs with a process pool (DOCKER_SUBMISSION.md
advises OCR-heavy pipelines to do this): MIB_WORKERS worker processes (default 4,
= the scored vCPU count), each running OCR single-threaded so they don't
oversubscribe. Set MIB_WORKERS=1 for a plain sequential run.

Runtime discipline (matches the read-only scoring container):
  - final output goes to argv[2] (never hardcoded); its parent is /output (writable)
  - no scratch files are written (OCR is in-memory), so a read-only root is fine
  - one bad PDF is skipped, never crashes the run: omitting a case costs only the
    small missing-case penalty, which beats emitting a garbage row
  - debug artifacts are written next to the output ONLY when MIB_DEBUG is set
"""
import csv
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path


def _safe_date(v):
    """Guarantee a schema-valid `arrival_date`. Any non-calendar value (e.g. an
    OCR misread like 2026-06-31) falls back to the missing-date placeholder, so
    validate_submission.py never hard-fails on the whole submission."""
    try:
        datetime.date.fromisoformat(str(v))
        return v
    except (ValueError, TypeError):
        return "1900-01-01"

_HERE = Path(__file__).resolve().parent

# Order matches schemas/submission.schema.json.
FIELDNAMES = ["case_id", "applicant_name", "species_code", "home_world", "visa_class",
              "sponsor_id", "arrival_date", "declared_purpose", "risk_flags", "fee_status",
              "adjudication", "confidence"]
CONTENT_FIELDS = FIELDNAMES[1:10]

DEBUG = bool(os.environ.get("MIB_DEBUG"))
WORKERS = max(1, int(os.environ.get("MIB_WORKERS", "4")))

_STATE = {}   # per-process: OCR engine + loaded classifier (built in the worker)


def _init_worker():
    """Import the heavy stack ONCE per process and cap OCR to a single thread."""
    os.environ.setdefault("MIB_OCR_THREADS", "1")   # 1 thread/worker; workers give the parallelism
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings
    # ExtraTrees can't nest loky parallelism inside a worker process; it correctly
    # falls back to n_jobs=1 (what we want). Silence the informational warning.
    warnings.filterwarnings("ignore", message="Loky-backed parallel loops")
    sys.path.insert(0, str(_HERE / "src"))
    sys.path.insert(0, str(_HERE / "src" / "pipeline"))
    import ocr, extract_v2, clf_final
    _STATE["ocr"] = ocr
    _STATE["extract"] = extract_v2
    _STATE["clf"] = clf_final.AdjudicatorFinal.load()


def _run_one(pdf_str: str) -> dict:
    """Full pipeline for one PDF. Returns {row|None, pages, runtime_s, case_id}."""
    pdf = Path(pdf_str)
    t0 = time.perf_counter()
    try:
        rec = _STATE["ocr"].ocr_document(pdf)
        fields = _STATE["extract"].extract_fields_v2(rec)
        feats = _STATE["extract"].extract_features(rec, fields)
        adjudication, confidence = _STATE["clf"].predict_one(rec, fields, feats)
        row = {"case_id": pdf.stem, **{k: fields.get(k) for k in CONTENT_FIELDS},
               "adjudication": adjudication, "confidence": round(float(confidence), 4)}
        row["arrival_date"] = _safe_date(row.get("arrival_date"))   # never emit an invalid date
        pages = len(rec.get("pages", []))
    except Exception as exc:  # noqa: BLE001 - never let one bad PDF kill the run
        print(f"[warn] skipping {pdf.name}: {exc}", file=sys.stderr)
        row, pages = None, None
    return {"row": row, "pages": pages, "case_id": pdf.stem,
            "runtime_s": round(time.perf_counter() - t0, 4)}


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, rows):
    with open(path, "w") as f:
        json.dump({"predictions": rows}, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(input_dir: str, output_path: str) -> None:
    pdfs = [str(p) for p in sorted(Path(input_dir).glob("*.pdf"))]

    if WORKERS == 1 or len(pdfs) <= 1:
        _init_worker()
        results = [_run_one(p) for p in pdfs]
    else:
        # spawn: parent never imports onnxruntime, so this is fork-safe and clean.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(WORKERS, len(pdfs)), initializer=_init_worker) as pool:
            results = pool.map(_run_one, pdfs, chunksize=1)

    rows = [r["row"] for r in results if r["row"] is not None]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    (write_jsonl if suffix == ".jsonl" else write_json if suffix == ".json" else write_csv)(output, rows)
    print(f"[info] wrote {len(rows)}/{len(pdfs)} predictions to {output}", file=sys.stderr)

    total = sum(r["runtime_s"] for r in results)
    if results:
        # wall-clock is what the budget is measured on; report both.
        slowest = max(results, key=lambda r: r["runtime_s"])
        print(f"[stats] {len(pdfs)} pdfs, {WORKERS} workers | sum-cpu {total:.1f}s | "
              f"slowest {slowest['case_id']} {slowest['runtime_s']}s", file=sys.stderr)

    if DEBUG:
        with open(output.parent / "stats.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps({k: r[k] for k in ("case_id", "pages", "runtime_s")},
                                   sort_keys=True) + "\n")
        print(f"[debug] wrote stats.jsonl ({len(results)} rows)", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: solution.py <input_pdf_dir> <output_path>")
    main(sys.argv[1], sys.argv[2])
