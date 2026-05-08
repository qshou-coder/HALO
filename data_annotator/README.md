# RoboTwin2.0 Dataset Annotation Pipeline

This repository provides tools for downloading, preprocessing, and annotating the **RoboTwin2.0** robot manipulation dataset using both **atomic action labeling** and **subtask-level semantic annotation** with **Qwen3-VL**.

---

## 📦 1. Download Dataset

Use `huggingface-cli` to download specific task data from the Hugging Face dataset repository:

```bash
huggingface-cli download \
  --repo-type dataset \
  TianxingChen/RoboTwin2.0 \
  dataset/stack_bowls_two/aloha-agilex_clean_50.zip \
  --local-dir ./
```

> Replace the file path with your desired task and robot variant (e.g., `beat_block_hammer/arx-x5_clean_50.zip`).

---

## 🔧 2. Atomic Action Annotation

### Batch Annotation

Run the following script to annotate entire tasks:

```bash
pip install -r requirements.txt
```

```bash
bash process_annotation.sh --task_name blocks_ranking_rgb/aloha-agilex_clean_50 --config ./configs/config.yaml
```

- `--task_name`: Required. Format: `<task>/<robot_variant>`
- `--config`: Optional. Defaults to `./configs/config_<task>_<robot>.yaml` if not specified.

### Directory Structure After Annotation

For a task like `adust_bottle/aloha-agilex_clean_50`, the following directories are generated:

```
./dataset/adust_bottle/aloha-agilex_clean_50/
├── endpose_data/          # Raw end-effector pose trajectories (.json)
├── label_data/            # Atomic action labels (.json)
└── label_vis/             # Visualization of labels (.png)
```

### Convert Pose Data 

Convert raw pose data (e.g., from HDF5 to JSON):

```bash
python convert.py --hdf5_path <path> --json_path <output_path>
```

### Annotate a Single Episode

```bash
python label.py \
  --config ./configs/config.yaml \
  --input_file ./dataset/place_object_basket/ur5_clean_50/endpose_data/episode0.json \
  --output_file ./test.json
```

### Visualize a Single Annotation

```bash
python visualize_labels.py --input ./test.json --output ./test.png
```


---

## 🧠 3. Subtask-Level Semantic Annotation (with Qwen3-VL)

### 3.1 Set Up Qwen3-VL Backend Service

#### Install Dependencies

```bash
pip install "vllm>=0.11.0"
pip install "qwen-vl-utils==0.0.14"
pip install modelscope
```

> Ensure your system meets GPU and CUDA requirements for `vLLM`.

#### Download Model Weights

```bash
modelscope download \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --local_dir Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

#### Launch API Server

```bash
nohup vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 18000 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs 8 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  > vllm_qwen3vl.log 2>&1 &
```

#### Monitor Logs

```bash
tail -f vllm_qwen3vl.log
```

The service will be available at `http://localhost:18000/v1`.

---

### 3.2 Run Subtask Annotation

#### Annotate All Tasks

```bash
bash run_all_task.sh
```

#### Annotate a Single Task

```bash
bash process_annotation_complete.sh --task_name beat_block_hammer/aloha-agilex_clean_50
```

#### Annotate a Single Episode

```bash
python main.py \
  --annotation_json ./dataset/beat_block_hammer/aloha-agilex_clean_50/label_data/episode1.json \
  --image_dir ./dataset/beat_block_hammer/aloha-agilex_clean_50/frame_images/episode1 \
  --output_json ./dataset/beat_block_hammer/aloha-agilex_clean_50/complete_annotation_data/episode1.json \
  --qwen_api_base http://localhost:18000/v1 \
  --qwen_model_name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --instruction_file ./dataset/beat_block_hammer/aloha-agilex_clean_50/instructions/episode1.json \
  --trajectory_image_path ./dataset/beat_block_hammer/aloha-agilex_clean_50/label_vis/episode1.png \
  --log_level INFO
```

#### Annotate Scene-Level Metadata (Optional)

```bash
python src/scene_annotator.py \
  --annotation_json ./dataset/beat_block_hammer/aloha-agilex_clean_50/complete_annotation_data/episode0.json \
  --image_dir ./dataset/beat_block_hammer/aloha-agilex_clean_50/frame_images/episode0 \
  --output_hdf5 ./output/episode0.hdf5 \
  --qwen_api_base http://localhost:18000/v1 \
  --qwen_model_name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --instruction_file ./dataset/beat_block_hammer/aloha-agilex_clean_50/instructions/episode0.json \
  --log_level INFO
```

> ⚠️ Note: Ensure paths match your local dataset structure.

---

## 📁 Dataset Structure Overview

```
dataset/
└── <task_name>/
    └── <robot_variant>/
        ├── endpose_data/          # Raw trajectory poses
        ├── frame_images/          # RGB frames per episode
        ├── instructions/          # Task instructions per episode
        ├── label_data/            # Atomic action labels
        ├── label_vis/             # Label visualizations
        └── complete_annotation_data/  # Subtask-level semantic annotations (generated)
```

---

## 🛠️ Requirements

- Python ≥ 3.9
- `huggingface_hub`, `numpy`, `opencv-python`, `Pillow`, `h5py`
- `vLLM` ≥ 0.11.0 (for Qwen3-VL)
- GPU with ≥ 80 GB VRAM (for Qwen3-VL-30B)

---

## 📜 License

This dataset and code are released under the [MIT License](LICENSE). Model weights are subject to the [Qwen license](https://github.com/QwenLM/Qwen/blob/main/LICENSE).

---

## 🙏 Acknowledgements

- [RoboTwin2.0 Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8) by Alibaba
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference

--- 