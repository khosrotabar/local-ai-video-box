#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# ============================================================
# LOCAL AI VIDEO BOX
# Recreates the tested 1x RTX 5090 AI Movie backend.
#
# Current scope:
#   - LTX-2.5 22B Distilled
#   - Wan 2.2 A14B NVFP4 / LightX2V
#   - SkyReels V2 DF 14B FP8
#   - FastAPI multi-engine API
#   - SQLite jobs
#   - API key auth
#   - real progress parsing
#   - cancel / process-tree termination
#
# Not included yet:
#   - image/reference upload
#   - long movie orchestration
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="${AI_MOVIE_ROOT:-/opt/ai-movie}"

ENGINES="$ROOT/engines"
MODELS="$ROOT/models"
CACHE="$ROOT/cache"
OUTPUTS="$ROOT/outputs"
LOGS="$ROOT/logs"
TEMP="$ROOT/temp"
STATE="$ROOT/.state"
SERVER="$ROOT/server"

LTX="$ENGINES/ltx/LTX-2"

WAN_ROOT="$ENGINES/wan"
WAN="$WAN_ROOT/LightX2V"
SAGE="$WAN_ROOT/SageAttention"
SPARGE="$WAN_ROOT/SpargeAttn"
CUTLASS="$WAN_ROOT/cutlass"

SKY="$ENGINES/skyreels-diffusers"

LTX_MODEL="$MODELS/ltx-2.5"
WAN_BASE="$MODELS/wan2.2-t2v-base"
WAN_QUANT="$MODELS/lightwan2.2-a14b-nvfp4"

HF_HOME="$CACHE/huggingface"

# ------------------------------------------------------------
# Fallback pins.
# pins.env copied from the working machine overrides these.
# ------------------------------------------------------------

LTX_REF_DEFAULT="a95ab856bf29407b6b066ede0abe1846050db56c"
LIGHTX2V_REF_DEFAULT="bb964f8a125ab5147b4645ec6895b727e67f37ba"
SAGE_REF_DEFAULT="d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
SPARGE_REF_DEFAULT="9c3886beb1077927d18ddb3f4f63ddde5964a513"
CUTLASS_REF_DEFAULT="59e3a3338d516ca6ce0e073af8da65289678a35c"

if [[ -f "$REPO_ROOT/pins.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/pins.env"
fi

LTX_REF="${LTX_REF:-$LTX_REF_DEFAULT}"
LIGHTX2V_REF="${LIGHTX2V_REF:-$LIGHTX2V_REF_DEFAULT}"
SAGE_REF="${SAGE_REF:-$SAGE_REF_DEFAULT}"
SPARGE_REF="${SPARGE_REF:-$SPARGE_REF_DEFAULT}"
CUTLASS_REF="${CUTLASS_REF:-$CUTLASS_REF_DEFAULT}"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

log() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

die() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    echo
    echo "BOOTSTRAP FAILED ❌"
    echo "Line: ${BASH_LINENO[0]}"
    echo "Exit code: $exit_code"
    exit "$exit_code"
}

trap on_error ERR

clone_pinned() {
    local url="$1"
    local dir="$2"
    local ref="$3"

    if [[ ! -d "$dir/.git" ]]; then
        mkdir -p "$(dirname "$dir")"
        git clone --filter=blob:none "$url" "$dir"
    fi

    git -C "$dir" fetch --depth 1 origin "$ref" || \
        git -C "$dir" fetch origin "$ref"

    git -C "$dir" checkout --detach FETCH_HEAD

    echo "$(basename "$dir"): $(git -C "$dir" rev-parse HEAD)"
}

# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

if [[ "${EUID}" -ne 0 ]]; then
    die "Run bootstrap as root."
fi

log "LOCAL AI VIDEO BOX BOOTSTRAP"

echo "Install root: $ROOT"

# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

log "CHECK GPU"

command -v nvidia-smi >/dev/null 2>&1 || \
    die "nvidia-smi is not available. Install/provide NVIDIA driver first."

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

echo "GPU: $GPU_NAME"

if [[ "$GPU_NAME" != *"RTX 5090"* ]]; then
    if [[ "${ALLOW_UNTESTED_GPU:-0}" != "1" ]]; then
        die "This bootstrap is currently validated for RTX 5090. Set ALLOW_UNTESTED_GPU=1 to override."
    fi

    echo "WARNING: continuing on an untested GPU."
fi

nvidia-smi

# ------------------------------------------------------------
# OS
# ------------------------------------------------------------

log "CHECK OPERATING SYSTEM"

source /etc/os-release

echo "OS: $PRETTY_NAME"

if [[ "${ID:-}" != "ubuntu" ]]; then
    die "This bootstrap currently targets Ubuntu."
fi

# ------------------------------------------------------------
# Disk
# ------------------------------------------------------------

mkdir -p "$ROOT"

AVAILABLE_KB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

echo "Free disk: ${AVAILABLE_GB} GB"

if (( AVAILABLE_GB < 180 )); then
    die "At least ~180GB free space is recommended."
fi

# ------------------------------------------------------------
# Base packages
# ------------------------------------------------------------

log "INSTALL SYSTEM DEPENDENCIES"

export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    ffmpeg \
    jq \
    tmux \
    htop \
    unzip \
    psmisc \
    iproute2 \
    openssl \
    libgl1 \
    libglib2.0-0

git lfs install

# ------------------------------------------------------------
# uv
# ------------------------------------------------------------

log "INSTALL UV"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"

command -v uv >/dev/null 2>&1 || \
    die "uv installation failed."

uv --version

# ------------------------------------------------------------
# CUDA Toolkit
#
# Do NOT touch the NVIDIA driver.
# We install only the CUDA 13.0 development toolkit if required.
# ------------------------------------------------------------

log "ENSURE CUDA 13.0 TOOLKIT"

if [[ ! -x /usr/local/cuda-13.0/bin/nvcc ]]; then

    if [[ "${VERSION_ID:-}" != "24.04" ]]; then
        die "Automatic CUDA toolkit installation currently expects Ubuntu 24.04."
    fi

    cd /tmp

    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
        wget -q \
          https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
          -O cuda-keyring_1.1-1_all.deb

        dpkg -i cuda-keyring_1.1-1_all.deb
    fi

    apt-get update
    apt-get install -y cuda-toolkit-13-0
fi

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$CUDA_HOME/bin/nvcc" --version

# ------------------------------------------------------------
# Directory structure
# ------------------------------------------------------------

log "CREATE DIRECTORY STRUCTURE"

mkdir -p \
    "$ENGINES/ltx" \
    "$WAN_ROOT" \
    "$MODELS" \
    "$OUTPUTS/api" \
    "$HF_HOME" \
    "$LOGS/api" \
    "$TEMP" \
    "$STATE" \
    "$SERVER"

chmod 700 "$SERVER"

# ------------------------------------------------------------
# Validate repo payload
# ------------------------------------------------------------

[[ -f "$REPO_ROOT/server/app.py" ]] || \
    die "Missing server/app.py"

[[ -f "$REPO_ROOT/server/skyreels_runner.py" ]] || \
    die "Missing server/skyreels_runner.py"

[[ -f "$REPO_ROOT/configs/wan_moe_t2v_5090.json" ]] || \
    die "Missing configs/wan_moe_t2v_5090.json"

cp "$REPO_ROOT/server/app.py" \
   "$SERVER/app.py"

cp "$REPO_ROOT/server/skyreels_runner.py" \
   "$SERVER/skyreels_runner.py"

# ------------------------------------------------------------
# Clone exact engine revisions
# ------------------------------------------------------------

log "CLONE PINNED ENGINE REVISIONS"

clone_pinned \
    https://github.com/Lightricks/LTX-2.git \
    "$LTX" \
    "$LTX_REF"

clone_pinned \
    https://github.com/ModelTC/LightX2V.git \
    "$WAN" \
    "$LIGHTX2V_REF"

clone_pinned \
    https://github.com/thu-ml/SageAttention.git \
    "$SAGE" \
    "$SAGE_REF"

clone_pinned \
    https://github.com/ModelTC/SpargeAttn.git \
    "$SPARGE" \
    "$SPARGE_REF"

clone_pinned \
    https://github.com/NVIDIA/cutlass.git \
    "$CUTLASS" \
    "$CUTLASS_REF"

# ============================================================
# LTX
# ============================================================

log "INSTALL LTX-2.5 ENVIRONMENT"

cd "$LTX"

uv sync

LTXPY="$LTX/.venv/bin/python"

"$LTXPY" - <<'PY'
import torch

print("LTX Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

assert torch.cuda.is_available()

print("LTX environment READY ✅")
PY

# ============================================================
# WAN / LightX2V
# ============================================================

log "INSTALL WAN / LIGHTX2V ENVIRONMENT"

if [[ ! -x "$WAN/.venv/bin/python" ]]; then
    uv venv \
        --python python3.12 \
        "$WAN/.venv"
fi

WANPY="$WAN/.venv/bin/python"

# Exact CUDA/PyTorch family used on the working RTX 5090.
uv pip install \
    --python "$WANPY" \
    --index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    torch==2.11.0+cu130 \
    torchvision==0.26.0+cu130 \
    torchaudio==2.11.0+cu130

# Install LightX2V package itself without allowing its dependency
# resolver to replace our CUDA-specific PyTorch.
uv pip install \
    --python "$WANPY" \
    --no-deps \
    -e "$WAN"

# Runtime dependencies needed by LightX2V inference.
uv pip install \
    --python "$WANPY" \
    packaging \
    ninja \
    cmake \
    scikit-build-core \
    wheel \
    numpy \
    scipy \
    diffusers \
    transformers \
    tokenizers \
    tqdm \
    accelerate \
    safetensors \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    einops \
    loguru \
    omegaconf \
    peft \
    qtorch \
    ftfy \
    aiohttp \
    pydantic \
    requests \
    decord \
    av \
    jsonschema \
    pymongo \
    modelscope \
    gradio \
    prometheus-client \
    gguf \
    PyJWT \
    fastapi \
    uvicorn \
    soundfile \
    pyzmq \
    "moviepy==1.0.3"

# FlashInfer RoPE — exact working package.
uv pip install \
    --python "$WANPY" \
    "flashinfer-python[cu13]==0.6.18.post1"

"$WANPY" - <<'PY'
import torch

print("WAN Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

assert torch.cuda.is_available()

print("WAN base environment READY ✅")
PY

# ------------------------------------------------------------
# LightX2V NVFP4 kernel
# ------------------------------------------------------------

log "BUILD LIGHTX2V NVFP4 KERNEL"

if [[ ! -f "$STATE/wan-nvfp4-kernel.ok" ]]; then

    cd "$WAN/lightx2v_kernel"

    source "$WAN/.venv/bin/activate"

    export MAX_JOBS="${MAX_JOBS:-16}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"

    rm -rf build dist

    uv build \
        --wheel \
        -Cbuild-dir=build \
        . \
        -Ccmake.define.CUTLASS_PATH="$CUTLASS" \
        --verbose \
        --color=always \
        --no-build-isolation

    uv pip install \
        --python "$WANPY" \
        --force-reinstall \
        --no-deps \
        dist/*.whl

    "$WANPY" test/nvfp4_nvfp4/test_bench2.py

    deactivate || true

    touch "$STATE/wan-nvfp4-kernel.ok"
else
    echo "NVFP4 kernel already built."
fi

# ------------------------------------------------------------
# SageAttention
# ------------------------------------------------------------

log "BUILD SAGEATTENTION FOR SM120"

if [[ ! -f "$STATE/sageattention-sm120.ok" ]]; then

    cd "$SAGE"

    export CUDA_HOME=/usr/local/cuda-13.0
    export CUDA_ARCHITECTURES="12.0"
    export TORCH_CUDA_ARCH_LIST="12.0"
    export MAX_JOBS="${MAX_JOBS:-16}"

    uv pip install \
        --python "$WANPY" \
        --no-build-isolation \
        --no-deps \
        -v \
        -e .

    "$WANPY" - <<'PY'
from sageattention import sageattn
print("SageAttention READY ✅")
PY

    touch "$STATE/sageattention-sm120.ok"
else
    echo "SageAttention already built."
fi

# ------------------------------------------------------------
# SpargeAttn
# ------------------------------------------------------------

log "BUILD SPARGEATTN FOR SM120"

if [[ ! -f "$STATE/spargeattn-sm120.ok" ]]; then

    cd "$SPARGE"

    export CUDA_HOME=/usr/local/cuda-13.0
    export TORCH_CUDA_ARCH_LIST="12.0"
    export MAX_JOBS="${MAX_JOBS:-16}"

    uv pip install \
        --python "$WANPY" \
        --no-build-isolation \
        --no-deps \
        -v \
        -e .

    "$WANPY" - <<'PY'
import spas_sage_attn
import spas_sage_attn._fused as fused
import spas_sage_attn._qattn as qattn

print("spas_sage_attn: OK")
print("_fused: OK")
print("_qattn: OK")
print("SpargeAttn READY ✅")
PY

    touch "$STATE/spargeattn-sm120.ok"
else
    echo "SpargeAttn already built."
fi

# ============================================================
# SKYREELS
# ============================================================

log "INSTALL SKYREELS DIFFUSERS ENVIRONMENT"

mkdir -p "$SKY"

if [[ ! -x "$SKY/.venv/bin/python" ]]; then
    uv venv \
        --python python3.12 \
        "$SKY/.venv"
fi

SKYPY="$SKY/.venv/bin/python"

uv pip install \
    --python "$SKYPY" \
    --index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    torch==2.14.0+cu130 \
    torchvision==0.29.0+cu130

uv pip install \
    --python "$SKYPY" \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface-hub \
    sentencepiece \
    protobuf \
    ftfy \
    imageio \
    imageio-ffmpeg \
    pillow

# Keep the exact TorchAO family that passed on the working server.
uv pip install \
    --python "$SKYPY" \
    --no-deps \
    torchao==0.18.0

"$SKYPY" - <<'PY'
import torch
import torchvision

from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingPipeline,
)

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("SkyReels Diffusers import: OK")
print("TorchAO import: OK")
print("SkyReels environment READY ✅")
PY

# ============================================================
# HUGGING FACE DOWNLOAD TOOL
# ============================================================

log "PREPARE MODEL DOWNLOADER"

TOOLS="$ROOT/tools"

if [[ ! -x "$TOOLS/.venv/bin/python" ]]; then
    mkdir -p "$TOOLS"

    uv venv \
        --python python3.12 \
        "$TOOLS/.venv"
fi

TOOLPY="$TOOLS/.venv/bin/python"

uv pip install \
    --python "$TOOLPY" \
    "huggingface_hub[hf_xet]"

# ============================================================
# ASK FOR HF TOKEN
# ============================================================

log "HUGGING FACE AUTHENTICATION"

echo "LTX-2.5 is gated."
echo "Make sure your Hugging Face account has already accepted its access terms."
echo
echo "The token will NOT be written to disk."
echo "The token will NOT be stored in the runtime .env."
echo

if [[ -z "${HF_TOKEN:-}" ]]; then
    read -r -s -p "Hugging Face token: " HF_TOKEN
    echo
fi

[[ -n "$HF_TOKEN" ]] || \
    die "Hugging Face token cannot be empty."

export HF_TOKEN
export HF_HOME
export HF_XET_HIGH_PERFORMANCE=1

# ============================================================
# DOWNLOAD MODELS
# ============================================================

log "DOWNLOAD LTX-2.5"

"$TOOLPY" - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lightricks/LTX-2.5",
    token=os.environ["HF_TOKEN"],
    local_dir="$LTX_MODEL",
    allow_patterns=[
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "model_patches/ltx-2.5-duration-head-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ],
)

print("LTX-2.5 MODEL READY ✅")
PY

log "DOWNLOAD WAN 2.2 BASE COMPONENTS"

"$TOOLPY" - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.2-T2V-A14B",
    token=os.environ["HF_TOKEN"],
    local_dir="$WAN_BASE",
    allow_patterns=[
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "configuration.json",
        "google/*",
        "low_noise_model/config.json",
    ],
)

print("WAN BASE COMPONENTS READY ✅")
PY

log "DOWNLOAD WAN 2.2 NVFP4 EXPERTS"

"$TOOLPY" - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lightx2v/LightWan2.2-A14B",
    token=os.environ["HF_TOKEN"],
    local_dir="$WAN_QUANT",
    allow_patterns=[
        "Wan2.2-T2V-A14B_NVFP4_Sparse_high.safetensors",
        "Wan2.2-T2V-A14B_NVFP4_Sparse_low.safetensors",
    ],
)

print("WAN NVFP4 EXPERTS READY ✅")
PY

log "DOWNLOAD SKYREELS V2 DF 14B"

"$TOOLPY" - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Skywork/SkyReels-V2-DF-14B-540P-Diffusers",
    token=os.environ["HF_TOKEN"],
    cache_dir="$HF_HOME",
)

print("SKYREELS MODEL READY ✅")
PY

# Remove token immediately after download phase.
unset HF_TOKEN

# ============================================================
# VERIFY LTX MODEL FILES
# ============================================================

log "VERIFY MODEL FILES"

test -f \
"$LTX_MODEL/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"

test -f \
"$LTX_MODEL/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

test -f \
"$LTX_MODEL/vae/ltx-2.5-video-vae-bf16.safetensors"

test -f \
"$LTX_MODEL/vae/ltx-2.5-audio-vae-bf16.safetensors"

test -f \
"$LTX_MODEL/model_patches/ltx-2.5-duration-head-bf16.safetensors"

test -f \
"$LTX_MODEL/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

test -f \
"$WAN_BASE/Wan2.1_VAE.pth"

test -f \
"$WAN_BASE/models_t5_umt5-xxl-enc-bf16.pth"

test -f \
"$WAN_BASE/low_noise_model/config.json"

test -f \
"$WAN_QUANT/Wan2.2-T2V-A14B_NVFP4_Sparse_high.safetensors"

test -f \
"$WAN_QUANT/Wan2.2-T2V-A14B_NVFP4_Sparse_low.safetensors"

echo "MODEL FILE CHECK READY ✅"

# ============================================================
# WAN CONFIG
# ============================================================

log "INSTALL WAN RTX 5090 CONFIG"

WAN_CONFIG_DIR="$WAN/configs/wan22/extreme"
WAN_CONFIG="$WAN_CONFIG_DIR/wan_moe_t2v_5090.json"

mkdir -p "$WAN_CONFIG_DIR"

cp \
    "$REPO_ROOT/configs/wan_moe_t2v_5090.json" \
    "$WAN_CONFIG"

TMP_CONFIG="$(mktemp)"

jq \
  --arg hi "$WAN_QUANT/Wan2.2-T2V-A14B_NVFP4_Sparse_high.safetensors" \
  --arg lo "$WAN_QUANT/Wan2.2-T2V-A14B_NVFP4_Sparse_low.safetensors" \
  '
    .high_noise_quantized_ckpt = $hi
    | .low_noise_quantized_ckpt = $lo
    | .high_noise_original_ckpt = null
    | .low_noise_original_ckpt = null
  ' \
  "$WAN_CONFIG" > "$TMP_CONFIG"

mv "$TMP_CONFIG" "$WAN_CONFIG"

jq '{
    infer_steps,
    target_video_length,
    target_height,
    target_width,
    dit_quant_scheme,
    high_noise_quantized_ckpt,
    low_noise_quantized_ckpt
}' "$WAN_CONFIG"

# ============================================================
# FINAL WAN IMPORT VERIFICATION
# ============================================================

log "VERIFY WAN RUNTIME"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$WANPY" - <<'PY'
import torch

from flashinfer.rope import (
    apply_rope_with_cos_sin_cache_inplace,
)

from sageattention import sageattn

import spas_sage_attn
import spas_sage_attn._fused as fused
import spas_sage_attn._qattn as qattn

import lightx2v.infer

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("CC:", torch.cuda.get_device_capability(0))

print("FlashInfer: OK")
print("SageAttention: OK")
print("SpargeAttn: OK")
print("LightX2V infer: OK")

print("WAN 2.2 A14B RUNTIME READY ✅")
PY

# ============================================================
# SERVER ENVIRONMENT
# ============================================================

log "INSTALL FASTAPI SERVER"

if [[ ! -x "$SERVER/.venv/bin/python" ]]; then
    uv venv \
        --python python3.12 \
        "$SERVER/.venv"
fi

SERVERPY="$SERVER/.venv/bin/python"

uv pip install \
    --python "$SERVERPY" \
    fastapi \
    "uvicorn[standard]" \
    pydantic

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

if [[ ! -f "$SERVER/.env" ]]; then

    API_KEY="$(openssl rand -hex 32)"

    cat > "$SERVER/.env" <<EOF
AI_MOVIE_API_KEY=$API_KEY
AI_MOVIE_PORT=11434
EOF

    chmod 600 "$SERVER/.env"
fi

# ------------------------------------------------------------
# Runtime launcher
# ------------------------------------------------------------

cat > "$SERVER/run.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/ai-movie/server

set -a
source .env
set +a

export HF_HOME=/opt/ai-movie/cache/huggingface
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

exec .venv/bin/uvicorn \
    app:app \
    --host 0.0.0.0 \
    --port "${AI_MOVIE_PORT:-11434}" \
    --workers 1
SH

chmod +x "$SERVER/run.sh"

# ============================================================
# PYTHON SYNTAX CHECK
# ============================================================

log "VERIFY BACKEND CODE"

"$SERVERPY" -m py_compile \
    "$SERVER/app.py" \
    "$SERVER/skyreels_runner.py"

echo "Backend syntax READY ✅"

# ============================================================
# START SERVER
# ============================================================

log "START AI MOVIE API"

tmux kill-session \
    -t ai-movie-api \
    2>/dev/null || true

tmux new-session \
    -d \
    -s ai-movie-api \
    "$SERVER/run.sh"

echo "Waiting for API..."

API_READY=0

for _ in $(seq 1 60); do

    if curl -fsS \
        http://127.0.0.1:11434/api/health \
        >/dev/null 2>&1
    then
        API_READY=1
        break
    fi

    sleep 1
done

if [[ "$API_READY" != "1" ]]; then

    echo
    echo "API failed to start."
    echo

    tmux capture-pane \
        -pt ai-movie-api \
        -S -150 || true

    die "FastAPI startup failed."
fi

# ============================================================
# VERIFY API
# ============================================================

log "VERIFY API"

curl -fsS \
    http://127.0.0.1:11434/api/health \
    | jq .

set -a
source "$SERVER/.env"
set +a

curl -fsS \
    http://127.0.0.1:11434/api/engines \
    -H "Authorization: Bearer $AI_MOVIE_API_KEY" \
    | jq .

# ============================================================
# FINAL
# ============================================================

log "BOOTSTRAP COMPLETE ✅"

echo
echo "Internal API:"
echo "  http://127.0.0.1:11434"
echo

echo "API KEY:"
grep '^AI_MOVIE_API_KEY=' "$SERVER/.env"

echo
echo "Models:"
echo "  LTX-2.5 22B Distilled"
echo "  Wan 2.2 A14B NVFP4"
echo "  SkyReels V2 DF 14B FP8"

echo
echo "Features:"
echo "  Multi-engine API        ✅"
echo "  SQLite job queue        ✅"
echo "  API key auth            ✅"
echo "  Real progress           ✅"
echo "  Cancellation            ✅"
echo "  GPU process cleanup     ✅"
echo "  Reference image input   ❌ tomorrow"
echo "  Long movie orchestrator ❌ later"

echo
echo "LOCAL AI VIDEO BOX READY 🚀"