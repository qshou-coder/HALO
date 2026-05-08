#!/bin/bash

# Override via env var (HuggingFace repo or local path):
#   QWEN_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 bash process_annotation_complete.sh ...
: "${QWEN_MODEL_NAME:=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
: "${QWEN_API_BASE:=http://localhost:18000/v1}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name)
            task_name="$2"
            shift 2
            ;;
        --config)
            config_path_override="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --task_name <task_name> [--config <config_path>]"
            echo "Example: $0 --task_name adust_bottle/aloha-agilex_clean_50"
            echo "         $0 --task_name adust_bottle/aloha-agilex_clean_50 --config ./my_custom_config.yaml"
            exit 1
            ;;
    esac
done

# Require --task_name
if [[ -z "${task_name}" ]]; then
    echo "Usage: $0 --task_name <task_name> [--config <config_path>]"
    echo "Example: $0 --task_name adust_bottle/aloha-agilex_clean_50"
    echo "         $0 --task_name adust_bottle/aloha-agilex_clean_50 --config ./my_custom_config.yaml"
    exit 1
fi

task_dir="${task_name}"
raw_annotation_dir="${task_dir}/label_data"
complete_annotation_dir="${task_dir}/complete_annotation_data"
video_dir="${task_dir}/video"
frames_dir="${task_dir}/frame_images"
instruction_dir="${task_dir}/instructions"
trajectory_image_dir="${task_dir}/label_vis"

for dir in "${frames_dir}" "${complete_annotation_dir}"; do
    if [[ ! -d "${dir}" ]]; then
        mkdir -p "${dir}"
    fi
done

# Step 1: extract video frames for each annotated trajectory
for json_file in "${raw_annotation_dir}"/*.json; do
    if [[ ! -f "${json_file}" ]]; then
        continue
    fi

    basename_json=$(basename "${json_file}" .json)
    image_output_dir="${frames_dir}/${basename_json}"
    if [[ ! -d "${image_output_dir}" ]] || [[ -z "$(ls -A "${image_output_dir}" 2>/dev/null)" ]]; then
        python src/extract_frames.py "${video_dir}/${basename_json}.mp4" "${image_output_dir}"
    fi
done

# Step 2: run the VLM-driven subtask + scene annotation pipeline
for json_file in "${raw_annotation_dir}"/*.json; do
    if [[ ! -f "${json_file}" ]]; then
        continue
    fi

    basename_json=$(basename "${json_file}" .json)
    image_output_dir="${frames_dir}/${basename_json}"

    output_file="${complete_annotation_dir}/${basename_json}.json"
    instruction_file="${instruction_dir}/${basename_json}.json"
    trajectory_image_file="${trajectory_image_dir}/${basename_json}.png"

    python main.py \
        --annotation_json "${json_file}" \
        --image_dir "${image_output_dir}" \
        --output_json "${output_file}" \
        --qwen_api_base "${QWEN_API_BASE}" \
        --qwen_model_name "${QWEN_MODEL_NAME}" \
        --instruction_file "${instruction_file}" \
        --trajectory_image_path "${trajectory_image_file}" \
        --log_level INFO
done

echo "All done for task: ${task_name}"

# Example:
#   bash process_annotation_complete.sh --task_name ./dataset/lift_pot/aloha-agilex_clean_50
