<div align="center">

  <h1 style="margin: -18px 0 0; font-size: 1.8em;">
    HALO: A Unified Vision-Language-Action Model for<br/>Embodied Multimodal Chain-of-Thought Reasoning
  </h1>

  <p><b>Accepted by ICML 2026</b></p>

  <p><b>Mixture-of-Transformers VLA &nbsp;·&nbsp; Embodied Multimodal CoT &nbsp;·&nbsp; Visual Foresight + Textual Reasoning + Action</b></p>

  [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.21157)
  [![Pretrained_weight](https://img.shields.io/badge/Pretrained_weight-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/qshou-coder/halo_pt_weight)
  [![Finetuned_weight](https://img.shields.io/badge/Finetuned_weight-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/qshou-coder/halo_ft_weight)
  [![Pretraining_Data](https://img.shields.io/badge/Pretraining_Data-624AFF?style=for-the-badge&logo=alibabacloud&logoColor=white)](https://modelscope.cn/datasets/shou123/Pretrain_Data)
  [![RoboTwin_data](https://img.shields.io/badge/RoboTwin_data-624AFF?style=for-the-badge&logo=alibabacloud&logoColor=white)](https://www.modelscope.cn/datasets/shou123/unlabeled_robotwin_data)

  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

## 📑 Table of Contents

- [📖 Introduction](#-introduction)
- [🗺️ Overview](#%EF%B8%8F-overview)
- [🍭 Method Overview](#-method-overview)
- [📊 Main Results](#-main-results)
- [📁 Repository Layout](#-repository-layout)
- [🛠️ Prerequisites](#%EF%B8%8F-prerequisites)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [📦 Download Weights & Data](#-download-weights--data)
- [🏋️ Fine-tuning](#%EF%B8%8F-fine-tuning)
- [🚀 Inference](#-inference)
- [🚧 Open-source Plan](#-open-source-plan)
- [🙌 Acknowledgements](#-acknowledgements)
- [📑 Citation](#-citation)

---

## 📖 Introduction

**HALO** is a unified Vision-Language-Action (VLA) model that performs **Embodied Multimodal Chain-of-Thought (EM-CoT) reasoning**. Unlike standard VLAs that map perceptual inputs directly to motor commands, HALO follows a deliberate "**think — imagine — execute**" cognitive pathway: it first writes a textual reasoning trace and a sub-task plan, then predicts a visual subgoal image to ground the plan in pixel space, and finally generates an action chunk conditioned on that EM-CoT context.

This human-like decomposition is realized through a **Mixture-of-Transformers (MoT)** architecture in which three specialized experts — **Multimodal Understanding**, **Visual Generation**, and **Action Prediction** — share a single self-attention stack but keep independent FFN parameters. Each expert preserves its native generative workflow (autoregressive tokens for text, flow-matching for visuals and actions), avoiding the conflicts that arise when heterogeneous capabilities are forced into a single monolithic model.

To train HALO at scale we release: (i) an automatic **EM-CoT data synthesis pipeline** that converts raw robot trajectories into trajectories augmented with task plans, sub-task reasoning, and subgoal images via action-primitive matching and a VLM annotator; and (ii) a two-stage training recipe — a **versatile pre-training stage** on VQA + visual generation + action prediction, followed by an **EM-CoT–augmented fine-tuning stage** that injects multimodal reasoning while preserving general world knowledge.

On the **RoboTwin 2.0** benchmark HALO reaches **80.5%** average success rate (Easy) and **26.4%** (Hard, domain-randomized), surpassing the strong π₀ baseline by **+34.1** and **+10.1** points respectively, and showing markedly better generalization under aggressive environmental randomization.

---

## 🗺️ Overview

| Component | Path | Description |
|-----------|------|-------------|
| **Pre-training / Fine-tuning** | [`train/`](train/) | Single-entry trainer (`pretrain_unified_navit.py`) for both stages with FSDP + EMA |
| **Inference** | [`inference/`](inference/) | `HALO` policy class + `HaloInferencer` (think → imagine → act loop) |
| **Modeling** | [`modeling/`](modeling/) | BAGEL MoT backbone, Qwen2 expert blocks, SigLIP-NaViT ViT, Flux VAE |
| **Data** | [`data/`](data/) | Iterable dataset implementations, transforms, video/parquet utilities |
| **Configs** | [`data/configs/`](data/configs/) | YAML recipes for dataset mixing & sampling |
| **Scripts** | [`scripts/`](scripts/) | One-command install / download / fine-tune wrappers |
| **Pretrained weights** | [HuggingFace](https://huggingface.co/qshou-coder/halo_pt_weight) | Stage-1 versatile-pre-training EMA checkpoint |
| **Datasets** | [ModelScope](https://modelscope.cn/datasets/shou123/) | `Pretrain_Data` (LLaVA-NeXT) + `unlabeled_robotwin_data` |

### Workflow at a glance

```
   ┌──────────────────┐    ┌──────────────────┐    ┌────────────────────┐
   │ BAGEL-7B-MoT     │ ── │ Versatile        │ ── │ EM-CoT-Augmented   │ ──▶ HALO
   │ + Qwen2.5-1.5B   │    │ Pre-training     │    │ Fine-tuning        │
   │ + SigLIP-NaViT   │    │ (VQA + VG + AP)  │    │ (RoboTwin + VQA)   │
   └──────────────────┘    └──────────────────┘    └────────────────────┘
                                   │                          │
                                   ▼                          ▼
                            LLaVA-NeXT-779k         RoboTwin 2.0 trajectories
                            OXE robot data          + EM-CoT synthesis pipeline
                            SSv2 ego-centric video
```

---

## 🍭 Method Overview

**Unified MoT architecture.** HALO factorizes textual reasoning, visual generation, and action prediction into three specialized experts, all sharing a single self-attention stack. Modality switching is gated by special control tokens (`<think_start>`, `<vision_start>`, `<action_start>`, ...). A carefully structured attention mask enforces causal generation across modalities while permitting bidirectional attention within an image frame; noise tokens used for flow-matching are isolated from their ground-truth targets to prevent leakage.

**EM-CoT data pipeline.** Raw `(observation, instruction, action)` trajectories are converted into supervised EM-CoT data in three phases:
1. **Action-primitive extraction.** Continuous low-level actions are mapped to discrete primitives (e.g. `arm_down`, `gripper_close`) via rule-based matching on proprioception.
2. **VLM annotation.** A large-scale VLM (Qwen3-VL) consumes the primitive sequence + frames and emits a high-level task plan, sub-task decomposition, and per-sub-task textual reasoning.
3. **Subgoal selection.** The terminal frame of each sub-task is designated as the visual subgoal — sparse but high-signal supervision for the visual-generation expert.

**Two-stage training recipe.**
- **Stage 1 — Versatile pre-training** mixes (a) general VQA on LLaVA-NeXT-779k, (b) visual generation on OXE + SSv2 with future-frame prediction, and (c) imitation learning on OXE actions. The total loss is `0.25·L_CE + 0.5·L_MSE + L_L1`.
- **Stage 2 — EM-CoT-augmented fine-tuning** trains on `(text-reasoning, subgoal-image, action)` triples plus auxiliary VQA to prevent forgetting; the joint loss `L_r + L_ô + L_a` orchestrates the full think-imagine-act chain.

---

## 📊 Main Results

Average success rate on **RoboTwin 2.0** (50 manipulation tasks, 100 evaluations each, ✕ Easy/Clean and Hard/Domain-randomized):

| Method | Easy | Hard |
|---|:---:|:---:|
| Diffusion Policy | 28.0 | 0.6 |
| RDT-1B | 34.5 | 13.7 |
| π₀ | 46.4 | 16.3 |
| **HALO – w/o EM-CoT** | **75.3** | **21.2** |
| **HALO (full EM-CoT)** | **80.5** | **26.4** |

**Highlights.**
- **+34.1** Easy / **+10.1** Hard over π₀; **+26.0** Hard over Diffusion Policy.
- The non-EM-CoT variant alone beats the strongest prior baseline by **+28.9** Easy, showing the strength of the versatile pre-training foundation.
- Full EM-CoT brings **+5.2** Easy and **+5.2** Hard on top of that, with the largest relative gains under aggressive environmental randomization.

**EM-CoT ablation.** Removing visual subgoals drops Easy from 80.5 → 76.1; removing textual reasoning drops it to 77.8; removing both yields 75.3 — **textual and visual reasoning contribute independent and additive gains**.

**Pre-training-recipe ablation.** Dropping visual-generation data (`w/o V`) costs 17.1 Easy points; dropping VQA on top (`w/o V+T`) costs another 15.3; without any pre-training (`w/o V+T+A`) Hard collapses to 0.0. Each pre-training source provides measurable downstream value.

---

## 📁 Repository Layout

```
HALO/
├── train/
│   ├── pretrain_unified_navit.py    # single trainer entry point (pre-train + fine-tune)
│   ├── fsdp_utils.py                # FSDP wrap / EMA / checkpoint save & load
│   └── train_utils.py
│
├── inference/
│   ├── model.py                     # HALO policy class (env-paramerized weight loading)
│   ├── halo_inferencer.py           # think → imagine → act inference loop
│   └── inferencer.py                # base interleaved inferencer
│
├── modeling/
│   ├── bagel/                       # MoT backbone (Bagel + Qwen2-NaViT)
│   ├── qwen2/                       # Qwen2 tokenizer + decoder layers
│   ├── siglip/                      # SigLIP-NaViT vision encoder
│   ├── autoencoder.py               # Flux-style VAE
│   └── action_tokenizer.py
│
├── data/
│   ├── dataset_info.py              # DATASET_REGISTRY + DATASET_INFO (data dirs)
│   ├── unified_dataset.py           # webdataset / RLDS / OXE / SSv2 / OXE-inverse
│   ├── robotwin_dataset.py          # HDF5 RoboTwin reader
│   ├── realworld_dataset.py         # real-world trajectory reader
│   ├── rlds/                        # RLDS Open-X-Embodiment loader
│   ├── transforms.py video_utils.py parquet_utils.py ...
│   └── configs/robotwin.yaml        # dataset-mixing recipe used by fine-tuning
│
├── scripts/
│   ├── install.sh                   # apt deps + pip + flash-attn + dlimp
│   ├── download.sh                  # BAGEL / SigLIP / Qwen / pretrain weights
│   ├── download_data.sh             # Pretrain_Data + unlabeled_robotwin_data
│   └── finetuning/finetune_no_cot.sh
│
├── overwatch/                       # logging helpers
├── requirements.txt
├── .env.example                     # HALO_MODEL_DIR / HALO_CKPT_DIR / HALO_DATA_DIR / HALO_OUTPUT_DIR
└── TRAIN.md                         # extended training notes
```

---

## 🛠️ Prerequisites

| Component | Minimum |
|-----------|---------|
| Python    | 3.10+ |
| CUDA      | 12.1+ |
| PyTorch   | 2.5.1 (with CUDA) |
| GPU       | 1× A100-80G for fine-tuning the 4.5B HALO; 8× A100/H100 recommended for full pre-training |
| Disk      | ~150 GB for weights + datasets |

All paths in scripts and configs are driven by four environment variables:

| Variable | Default | Used for |
|---|---|---|
| `HALO_MODEL_DIR`  | `./downloads/pretrained_models` | BAGEL / Qwen / SigLIP weights |
| `HALO_CKPT_DIR`   | `./downloads/my_checkpoints`    | Released HALO checkpoints |
| `HALO_DATA_DIR`   | `./download_data`               | Training datasets |
| `HALO_OUTPUT_DIR` | `./outputs`                     | Logs, results, fine-tune checkpoints |

Copy `.env.example` to `.env` and adjust to your filesystem.

---

## ⚙️ Installation

```bash
git clone https://github.com/qshou-coder/HALO.git
cd HALO

# system + Python deps + flash-attn + dlimp + imgaug
bash scripts/install.sh
```

`install.sh` installs `requirements.txt`, `flash_attn==2.5.8`, `webdataset`, `imgaug`, `dlimp` (from GitHub), and `tensorflow_graphics`.

---

## 📦 Download Weights & Data

```bash
# Pretrained model weights (~30 GB):
#   BAGEL-7B-MoT (HF, ema.safetensors filtered)
#   SigLIP-so400m-14-980-flash-attn2-navit (HF)
#   Qwen_1.5B_model (ModelScope)
#   halo_pt_weight  ← Stage-1 versatile-pre-training EMA checkpoint
bash scripts/download.sh

# Training datasets (large, optional — only needed for fine-tuning):
#   shou123/Pretrain_Data           → LLaVA-NeXT-Data (general VQA)
#   shou123/unlabeled_robotwin_data → RoboTwin 2.0 trajectories (no EM-CoT labels)
bash scripts/download_data.sh
```

---

## 🏋️ Fine-tuning

A single-node 8-GPU launch script for the **w/o EM-CoT** RoboTwin fine-tune:

```bash
bash scripts/finetuning/finetune_no_cot.sh
```

Key knobs in `scripts/finetuning/finetune_no_cot.sh`:

| Flag | Default | Description |
|---|---|---|
| `--dataset_config_file` | `data/configs/robotwin.yaml` | Dataset-mixing recipe (RoboTwin + LLaVA-NeXT) |
| `--ckpt_path` | `${HALO_CKPT_DIR}/halo_pt_weight` | Stage-1 EMA weights to start from |
| `--max_num_tokens` | `21202` | Hard cap for packed batch token count |
| `--num_shard / --num_replicate` | `8 / 4` | FSDP HSDP topology |
| `--lr / --warmup_steps` | `5e-5 / 500` | Optimizer schedule |

Multi-node launches are supported via the standard torchrun env vars (`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, `GPUS_PER_NODE`); the script provides single-node fallbacks for all of them.

> Extended notes — full argument reference, mixing-recipe semantics, expected loss curves: [`TRAIN.md`](TRAIN.md).

---

## 🚀 Inference

The `HALO` class in [`inference/model.py`](inference/model.py) loads the model and runs the think-imagine-act loop end-to-end:

```python
from inference.model import HALO

policy = HALO(pretrained_model_path="path/to/your/finetuned/ema.safetensors")

# observations: list of RGB numpy arrays (history frames)
# instruction:  natural-language task description
action_chunk = policy.get_action(instruction, obs_window=observations)
```

Internally `HALO.get_action`:
1. resizes observations through the SigLIP-NaViT and VAE transforms,
2. samples a textual reasoning trace + sub-task plan with the Multimodal Understanding expert,
3. predicts a visual subgoal image with the Visual Generation expert (flow-matching, 50 steps),
4. predicts a `K=16` action chunk with the Action Prediction expert (flow-matching, 10 steps).

`use_subtask` and `use_goal_image` are enabled probabilistically (`use_prob=0.2`) to let the EM-CoT model gracefully fall back to the reactive policy when reasoning is not necessary.

---

## 🚧 Open-source Plan

| Item | Status |
|---|:---:|
| Fine-tuning code (w/o EM-CoT) | ✅ Released |
| Inference code | ✅ Released |
| Versatile-pre-training weights (`halo_pt_weight`) | ✅ Released |
| RoboTwin fine-tune weights (w/o EM-CoT) | ✅ Released |
| Fine-tuning datasets (`Pretrain_Data`, `unlabeled_robotwin_data`) | ✅ Released |
| EM-CoT data synthesis pipeline | 🚧 In progress |
| EM-CoT fine-tuning code | 🚧 In progress |
| RoboTwin fine-tune weights (with EM-CoT) | 🚧 In progress |
| Latest real-robot video demo | 🚧 In progress |

Stay tuned — the remaining items will be released **as soon as the internal review finishes**.

---

## 🙌 Acknowledgements

This repository builds on several outstanding open-source projects:

- [**BAGEL**](https://github.com/bytedance-seed/BAGEL) — Mixture-of-Transformers backbone we adopt and extend with an action-prediction expert.
- [**Qwen2.5**](https://github.com/QwenLM/Qwen2.5) — language-model expert initializer.
- [**SigLIP / NaViT**](https://github.com/google-research/big_vision) — vision encoder for the multimodal-understanding expert.
- [**Flux-VAE**](https://github.com/black-forest-labs/flux) — high-fidelity image autoencoder used for the visual-generation expert.
- [**Open-X-Embodiment**](https://robotics-transformer-x.github.io/), [**SSv2**](https://www.qualcomm.com/developer/software/something-something-v-2-dataset), [**LLaVA-NeXT**](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) — pre-training data sources.
- [**RoboTwin 2.0**](https://github.com/TianxingChen/RoboTwin) — bimanual manipulation benchmark used for fine-tuning and evaluation.
- [**dlimp**](https://github.com/kvablack/dlimp) — RLDS dataloader utilities.

We further thank the authors of CoT-VLA, MoT-VLA, and ManualVLA whose ideas inspired our embodied multimodal chain-of-thought design.

Project-specific code is released under the root [`LICENSE`](LICENSE) (Apache 2.0).

---

## 📑 Citation

If you find HALO useful in your research, please cite:

```bibtex
@article{shou2026halo,
  title  = {HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning},
  author = {Shou, Quanxin and Zhu, Fangqi and Chen, Shawn and Yan, Puxin and Yan, Zhengyang and
            Miao, Yikun and Pang, Xiaoyi and Hong, Zicong and Shi, Ruikai and Huang, Hao and
            Zhang, Jie and Guo, Song},
  journal = {arXiv preprint arXiv:2602.21157},
  year   = {2026}
}
```
