FROM python:3.12-slim

WORKDIR /app

# OpenCV (a transitive dependency of rapidocr-onnxruntime) needs these two shared
# libraries. Nothing else requires system packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first so this layer caches across code changes.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Application code + model artifacts (co-located under src/pipeline/ so each module
# loads its artifact relative to its own file).
COPY run.sh solution.py /app/
COPY src/ /app/src/
RUN chmod +x /app/run.sh

# 8090 calls the image as:  <image> /input /output/predictions.jsonl
ENTRYPOINT ["/app/run.sh"]
