# ---- Challenge submission Dockerfile ----
# Build: docker build -t orena-frame-submission .
# The challenge runs: docker run --gpus all <image>
# No internet access inside container at evaluation time.

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
COPY submission/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source package
COPY src/ ./src/
RUN pip install -e ./src  2>/dev/null || true

# Copy submission entrypoint
COPY submission/predict.py .

# Copy cached model weights (add before submission — weights must be bundled,
# no internet at eval time)
# COPY model_weights/ ./model_weights/

# Entrypoint called by the challenge runner
ENTRYPOINT ["python", "predict.py"]
