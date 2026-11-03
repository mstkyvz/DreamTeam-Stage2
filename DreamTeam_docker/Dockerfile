# syntax=docker/dockerfile:1.7
# Base matches the box every reported number was measured on: nvidia/cuda 13.0 + pip vllm==0.28.0
# (which pulls torch 2.13.0+cu130). `devel`, not `runtime`, because Triton's runtime JIT needs
# ptxas/nvcc and the FP8 MoE gate goes through Triton on this GPU. The organizer's host driver is
# 580.126.20, which is CUDA-13.0 capable, so no cuda-compat layer is installed -- on drivers >=580
# a compat layer causes CUDA Error 803 rather than helping.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Official runs mount the rootfs READ-ONLY and give exactly one writable place: an 8 GB /tmp
# tmpfs (plus the bind-mounted output dir). Every cache any library might write must therefore
# live under /tmp, and every one of these has been observed to break a read-only run when unset.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    TRITON_CACHE_DIR=/tmp/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cache/torchinductor \
    VLLM_CACHE_ROOT=/tmp/cache/vllm \
    VLLM_CONFIG_ROOT=/tmp/cache/vllm_config \
    FLASHINFER_WORKSPACE_BASE=/tmp/cache/flashinfer \
    HF_HOME=/tmp/cache/huggingface \
    DT_ARTIFACT_DIR=/tmp/dt_artifacts \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0

# gcc/g++/python3-dev back Triton's runtime JIT (kernel launcher compilation). Without
# python3-dev the Triton path fails at import time, not at first use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ curl ca-certificates \
    && curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
        | tar -xz -C /usr/local/bin ngrok \
    && ngrok version \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# NOT INSTALLED, on purpose: `flashinfer-cubin` and `flashinfer-jit-cache`.
# They look like an obvious win -- prebuilt kernels, no JIT on first boot -- and one report during
# development placed a matching `0.6.16.post3+cu130` wheel on GitHub releases. It is not there:
# both URLs return 404, and no index carries a build matching flashinfer-python 0.6.16.post3
# (only cubin 0.6.13 and jit-cache 0.6.17). Installing the mismatched pair is WORSE than skipping
# them, because their version check is a hard import gate -- `import flashinfer` then raises and
# the champion silently loses its FlashInfer backend. Two independent 90,000-row runs measured the
# champion booting with `FlashInferImpl` active in 45-56 s without them, so the 215 s JIT boot
# they were meant to remove does not occur on this configuration at all.
RUN python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# The checkpoint is COPY'd from the build context: no runtime download, no build secret, and
# nothing to leak into a layer. --chown/--chmod on COPY avoids a duplicate multi-GB chown layer
# and keeps the files world-readable, which is required because official runs execute as an
# arbitrary host uid that has no account in this image.
# ONE checkpoint: lokeshe09/gemma-4-31B-it-INT8, 30.63 GiB.
# The development pipeline also supports a cheap MoE gate that escalates only the hardest rows to
# this model. Measured on the same 90,000 rows it is faster and worse at every fraction tried --
# 0.8741 @5%, 0.8798 @10%, 0.8903 @15%, <=0.8935 @20%, against champion-only's 0.9131 -- and it
# only draws level around 30% routing. The organizer allows 150 minutes and this finishes in 46.9,
# so there is nothing to buy with the time the cascade saves. The gate is not downloaded, not
# shipped, and not in this image.
# --chown/--chmod on COPY avoids a duplicate multi-GB chown layer and keeps the files
# world-readable, which is required because official runs execute as an arbitrary host uid.
COPY --chown=10001:10001 --chmod=755 models/champ_g31int8 /models/champ_g31int8

COPY --chown=10001:10001 --chmod=755 src ./src
COPY --chown=10001:10001 --chmod=755 entrypoint.sh ./entrypoint.sh

# A named account exists so an interactive `docker run` without --user still works; official runs
# override it with --user anyway, which is why nothing above depends on this uid.
RUN useradd --create-home --uid 10001 competitor
USER competitor

# Fail fast and locally if the image cannot even import its own entry points.
RUN python -c "import sys; sys.path.insert(0,'/app/src'); import app.config, app.wiring, services.cascade_service; print('import graph OK')"

ENTRYPOINT ["/app/entrypoint.sh"]
