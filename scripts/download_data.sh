#!/bin/bash
set -e
: "${HALO_DATA_DIR:=./download_data}"
mkdir -p "$HALO_DATA_DIR"

if ! git lfs version &>/dev/null; then
    echo "git-lfs not found, attempting to install..."
    sudo apt-get update && sudo apt-get install -y git-lfs || echo "Please install git-lfs manually (e.g. conda install git-lfs)"
fi
git lfs install

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

# ================= 1. Download Pretrain_Data (ModelScope) =================
echo "=== [1/2] Downloading dataset: Pretrain_Data (ModelScope) ==="
git_download "https://modelscope.cn/datasets/shou123/Pretrain_Data.git" "$HALO_DATA_DIR/Pretrain_Data"

# ================= 2. Download unlabeled_robotwin_data (ModelScope) =================
echo "=== [2/2] Downloading dataset: unlabeled_robotwin_data (ModelScope) ==="
git_download "https://modelscope.cn/datasets/shou123/unlabeled_robotwin_data.git" "$HALO_DATA_DIR/unlabeled_robotwin_data"

echo " "
echo "#######################################################"
echo "All datasets downloaded!"
echo "Data location: $HALO_DATA_DIR"
echo "#######################################################"
