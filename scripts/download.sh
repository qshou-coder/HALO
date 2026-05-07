#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
ROOT_DIR="./downloads"
MODEL_DIR="$ROOT_DIR/pretrained_models"
CKPT_DIR="$ROOT_DIR/my_checkpoints"

mkdir -p "$MODEL_DIR" "$CKPT_DIR"

echo "=== Installing required tools (modelscope, huggingface_hub) ==="
pip install -U modelscope huggingface_hub==0.34.0 -q
pip install datasets

if ! git lfs version &>/dev/null; then
    echo "git-lfs not found, attempting to install..."
    sudo apt-get update && sudo apt-get install -y git-lfs || echo "Please install git-lfs manually (e.g. conda install git-lfs)"
    git lfs install
else
    echo "git-lfs already installed"
    git lfs install
fi

function git_download() {
    URL=$1
    DIR=$2
    if [ ! -d "$DIR" ]; then
        echo "Cloning: $URL -> $DIR"
        git clone "$URL" "$DIR"
    else
        echo "Directory exists, pulling latest: $DIR"
        cd "$DIR"
        git pull
        git lfs pull
        cd - > /dev/null
    fi
}

# ================= 1. Download Bagel (HuggingFace, exclude ema.safetensors) =================
echo "=== [1/4] Downloading model: BAGEL-7B-MoT (HuggingFace) ==="
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT \
    --local-dir "$MODEL_DIR/BAGEL-7B-MoT" \
    --exclude "*ema.safetensors*"

# ================= 2. Download Siglip (HuggingFace) =================
echo "=== [2/4] Downloading model: Siglip (HuggingFace) ==="
huggingface-cli download HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit \
    --local-dir "$MODEL_DIR/siglip-so400m-14-980-flash-attn2-navit"

# ================= 3. Download Qwen 1.5B (ModelScope) =================
echo "=== [3/4] Downloading model: Qwen_1.5B_model (ModelScope) ==="
git_download "https://modelscope.cn/models/shou123/Qwen_1.5B_model.git" "$MODEL_DIR/Qwen_1.5B_model"

# ================= 4. Download Pretrained Weights (HuggingFace) =================
echo "=== [4/4] Downloading model: halo_pt_weight (HuggingFace) ==="
huggingface-cli download qshou-coder/halo_pt_weight \
    --local-dir "$CKPT_DIR/halo_pt_weight"

echo " "
echo "#######################################################"
echo "All downloads complete!"
echo "Models: $MODEL_DIR"
echo "Checkpoints: $CKPT_DIR"
echo "#######################################################"
