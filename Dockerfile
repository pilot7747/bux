# syntax=docker/dockerfile:1

# GPU image for bux: JAX on CUDA 13 for Blackwell (and Hopper), plus tokamax's Pallas kernels
# (FlashAttention, MoE grouped matmul) and cudnn-frontend's (KDA), which tokamax's extra brings
# (docs/training.md, Speed on GPUs). CUDA, cuDNN and NCCL come from the
# `cuda` extra's pip wheels, pinned in uv.lock, so the base image only needs to supply the driver
# plumbing; CUDA_VERSION should track the toolkit those wheels ship.
#
# Its entrypoint is the launcher, which takes a .bux file and trains what is left of its runs,
# a GPU each; with no file, it trains the smoke run. Mount data/ and runs/ to keep them:
#
#   docker build -t bux .
#   docker run --rm --gpus all bux
#   docker run --rm --gpus all -v "$PWD/data:/opt/bux/data" -v "$PWD/runs:/opt/bux/runs" \
#     bux examples/experiments/speedrun.bux
ARG CUDA_VERSION=13.4.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# Triton (behind tokamax's "triton" kernels) compiles its launcher with the host C compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# The CUDA 13 wheels work with any 580+ driver through minor version compatibility, so don't let
# the container toolkit demand a driver as new as the base image's toolkit.
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0"

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

# Create bux user with UID 1000 for k8s compatibility. Ubuntu images already ship `ubuntu` with
# that UID, so take it over (home included) rather than collide with it.
ARG USERNAME=bux
RUN if getent passwd 1000 >/dev/null 2>&1; then \
  existing_user=$(getent passwd 1000 | cut -d: -f1) && \
  usermod -l $USERNAME -d /home/$USERNAME -m $existing_user && \
  groupmod -n $USERNAME $existing_user; \
  else \
  useradd -u 1000 -m -s /bin/bash $USERNAME; \
  fi

# Python, the venv and the source live in /opt, owned by the user so it can `uv sync` more
# extras (e.g. the dev group) at runtime.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
  UV_PROJECT_ENVIRONMENT=/opt/venv \
  VIRTUAL_ENV=/opt/venv \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy
RUN install -d -o 1000 -g 1000 /opt/python /opt/venv /opt/bux

ENV PATH="/opt/venv/bin:${PATH}"

# `uv run` would otherwise re-sync to the default dependencies and drop the cuda extra.
ENV UV_NO_SYNC=1

# Numeric, so k8s can enforce runAsNonRoot without an explicit runAsUser.
USER 1000:1000
WORKDIR /opt/bux

# Dependencies first, so editing the source does not invalidate the install layer.
COPY --chown=1000:1000 .python-version pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/$USERNAME/.cache/uv,uid=1000,gid=1000 \
  uv sync --locked --no-dev --extra cuda --no-install-project

COPY --chown=1000:1000 . .
RUN --mount=type=cache,target=/home/$USERNAME/.cache/uv,uid=1000,gid=1000 \
  uv sync --locked --no-dev --extra cuda

# Exec form, so that the launcher is PID 1 and gets the SIGTERM that stops a container, at
# which it checkpoints the step it is at and exits.
ENTRYPOINT ["bux", "launch"]
CMD ["examples/experiments/smoke.bux"]
