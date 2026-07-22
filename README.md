# MIB Doc Challenge — Solution

An offline, CPU-only pipeline that reads each PDF packet and emits a structured
applicant record plus an adjudication (`APPROVED` / `DENIED` / `NEEDS_REVIEW`) with
a calibrated confidence. No network, no GPU, no external services at runtime.

**Held-out dev score: 124.07 / 150** (classification 63.3/80, extraction 43.8/50,
calibration 17.0/20, 0 catastrophic false approvals), at **~1.5 s/PDF** on 4 vCPUs.
See [`MEMO.md`](MEMO.md) for the approach, failure modes, and next steps.

## Run it

```bash
docker build -t mib-submission .

mkdir -p /tmp/mib-out
docker run --rm \
  --network none --cpus 4 --memory 8g --pids-limit 512 \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/tmp/mib-out,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

The image accepts `<input_pdf_dir> <output_predictions_path>` and writes one JSONL
object per answered case. `make build` and `make docker-test` wrap this.

## How it works

Per PDF: **OCR → field extraction → adjudication.**

1. **`src/ocr.py`** — render pages (pypdfium2, ~200 DPI), read them with RapidOCR
   (PP-OCR via ONNX; models bundled in the wheel, so fully offline). Orientation is
   *gated*: OCR as-rendered, and only rotate when a page doesn't look upright. The
   untrusted PDF text layer is captured separately (`hidden_text`) and never used as a
   value source.
2. **`src/pipeline/extract_v2.py`** (+ `vocab.py`, `repair.py`) — label-anchored
   extraction, then snap the closed-vocabulary fields to values mined from the training
   split (unseen values pass through unchanged). Spacing repair uses only visible OCR.
3. **`src/pipeline/flags.py`** — risk-flag detection (rules mined by distant supervision;
   precision 1.000 per flag).
4. **`src/pipeline/clf_final.py`** — a calibrated, cost-sensitive classifier (sklearn
   `HistGradientBoosting` + ExtraTrees blend). Confidence is the model's calibrated
   probability. **No feature is derived from the untrusted text layer**, so hidden-text
   injection cannot move an output.

The work is parallelized across the vCPUs (a process pool, each worker OCR-single-threaded).

## Layout

```
Dockerfile            offline image; entrypoint = run.sh
run.sh                <input_pdf_dir> <output_path>
solution.py           orchestrator: parallel loop -> predictions.jsonl
requirements.txt      pinned offline deps
MEMO.md               1–2 page technical memo
src/
  ocr.py              PDF -> OCR record (gated orientation)
  pipeline/           the extraction + classification modules, with artifacts
    extract_v2.py  vocab.py (+vocab_train.json)  repair.py  flags.py
    clf_final.py (+clf_final_model.joblib)  extract_v2 priors_train.json
    clf_d.py  clf_r2_1.py  clf_r2_3.py  common.py
```

Model artifacts are co-located with their modules (each loads relative to its own
file). Total artifacts ≈ 5.4 MiB — far under the 250 MiB / 1 GiB limits; image ≈ 0.3 GiB.

## Runtime notes

- **Offline**: RapidOCR's ONNX models and PDFium ship inside the pip wheels; nothing is
  fetched at run time. Verified under `--network none`.
- **Read-only root**: no scratch files are written (OCR is in-memory); only `/output` is
  written. Verified under `--read-only` with a tmpfs `/tmp`.
- **Budget**: ~1.5 s/PDF wall on 4 vCPUs (measured in a Docker VM where onnxruntime falls
  back to unoptimized ARM kernels; native x86 scoring hardware should be faster). Tune the
  worker count with `MIB_WORKERS` (default 4); `MIB_WORKERS=1` runs sequentially.
- **Determinism**: stable from a clean checkout on fixed hardware. A few borderline cases
  can flip across CPU/BLAS builds due to floating-point OCR differences; the aggregate
  score is stable (container 124.07 vs local 123.89 on the same sample).

## Local dev (inside the challenge repo)

```bash
python3 solution/solution.py data/train /tmp/out/predictions.jsonl
python3 scripts/evaluate.py --truth data/train_labels.csv \
  --submission /tmp/out/predictions.jsonl --output-json /tmp/out/evaluation.json
```
