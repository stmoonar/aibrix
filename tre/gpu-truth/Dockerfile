# Node GPU-truth agent (deploy/scripts/gpu_truth_agent.py): publishes per-GPU
# used/total memory read with nvidia-smi to tre:gpu_truth:<node> in TRE Redis.
#
# Build (context = tre/, like the other TRE components):
#   docker build -f gpu-truth/Dockerfile -t tre-v2-gpu-truth:<YYYYMMDD>-<sha> .
#
# No CUDA userspace is baked in. nvidia-smi and libnvidia-ml.so come from the
# host driver: the NVIDIA container runtime injects them into any container
# started with NVIDIA_VISIBLE_DEVICES set (the DaemonSet sets it) when the
# requested driver capabilities include "utility". The image therefore stays
# a plain Python base and does not pin a CUDA / driver version.
#
# BASE_IMAGE is a public upstream tag (python:3.11.11-slim-bookworm, digest
# sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5 at
# the time of writing); override it with --build-arg for an offline mirror.
ARG BASE_IMAGE=python:3.11.11-slim-bookworm
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=utility

COPY gpu-truth/requirements.txt /tmp/gpu-truth-requirements.txt
RUN pip install --no-cache-dir -r /tmp/gpu-truth-requirements.txt \
    && rm /tmp/gpu-truth-requirements.txt

COPY deploy/scripts/gpu_truth_agent.py /app/gpu_truth_agent.py

# Read-only NVML access works without root (/dev/nvidia* are world-accessible
# under the NVIDIA runtime); no passwd entry is needed for a numeric user.
USER 65532:65532
WORKDIR /app
ENTRYPOINT ["python3", "/app/gpu_truth_agent.py"]
