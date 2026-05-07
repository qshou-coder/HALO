: "${HALO_MODEL_DIR:=./downloads/pretrained_models}"
: "${HALO_CKPT_DIR:=./downloads/my_checkpoints}"
: "${HALO_DATA_DIR:=./download_data}"
: "${HALO_OUTPUT_DIR:=./outputs}"
# export WANDB_API_KEY=<your_wandb_key>  # see https://wandb.ai/authorize

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_HOST=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_addr=$MASTER_HOST \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --ckpt_path ${HALO_CKPT_DIR}/halo_pt_weight \
  --model_path ${HALO_MODEL_DIR}/BAGEL-7B-MoT \
  --llm_path ${HALO_MODEL_DIR}/Qwen_1.5B_model \
  --vit_path ${HALO_MODEL_DIR}/siglip-so400m-14-980-flash-attn2-navit \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --dataset_config_file ./data/configs/robotwin.yaml \
  --num_workers 4  \
  --max_num_tokens_per_sample 16384 \
  --max_num_tokens 21202 \
  --visual_gen False \
  --visual_und True \
  --action_gen True \
  --results_dir ${HALO_OUTPUT_DIR}/finetune_ckpt/naive \
  --log_dir ${HALO_OUTPUT_DIR}/log \
  --checkpoint_dir ${HALO_OUTPUT_DIR}/finetune_ckpt/naive \
  --wandb_project "Main_experiment" \
  --wandb_name "no_cot_try01" \
  --wandb_runid "1" \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema True \
  --log_every 20 \
  --save_every 2000 \
  --expected_num_tokens 19768 \
  --use_flex False \
  --num_shard 8 \
  --num_replicate 4 \
  --finetune_from_pretrained True \
  --lr 5e-5 \
  --warmup_steps 500