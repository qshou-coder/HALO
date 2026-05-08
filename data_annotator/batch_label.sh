#!/bin/bash

# ================= Configuration =================
# Override via env var:  BASE_DATASET_DIR=/path/to/dataset bash batch_label.sh ...
: "${BASE_DATASET_DIR:=./dataset}"
DEFAULT_CONFIG="config/config.yaml"
# ==================================================

config_path_override="${DEFAULT_CONFIG}"
task_names=()
auto_all=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                task_names+=("$1")
                shift
            done
            ;;
        --all)
            auto_all=true
            shift
            ;;
        --config)
            config_path_override="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--tasks cat1/task1 cat1/task2] [--all] [--config <path>]"
            exit 1
            ;;
    esac
done

# Auto-discover tasks: scan all second-level subdirectories
if [[ "${auto_all}" == true ]]; then
    echo "Scanning for all tasks in ${BASE_DATASET_DIR}/*/* ..."
    # find all dataset/category/task two-level directories
    while IFS= read -r dir; do
        # Only count it as a task if a 'data' folder exists inside
        if [[ -d "${dir}/data" ]]; then
            # Extract category/task relative path
            rel_path=$(echo "${dir}" | rev | cut -d'/' -f1,2 | rev)
            task_names+=("${rel_path}")
        fi
    done < <(find "${BASE_DATASET_DIR}" -mindepth 2 -maxdepth 2 -type d)
fi

# Sanity check: at least one task
if [[ ${#task_names[@]} -eq 0 ]]; then
    echo "Error: No tasks found. Check your path or use --all / --tasks."
    exit 1
fi

echo "Total tasks to process: ${#task_names[@]}"

# ================= Main loop =================
for task_name in "${task_names[@]}"; do
    task_dir="${BASE_DATASET_DIR}/${task_name}"

    echo ""
    echo "################################################################"
    echo "Processing Task: [ ${task_name} ]"
    echo "Path: ${task_dir}"
    echo "################################################################"

    # 1. Prepare directories
    endpose_dir="${task_dir}/endpose_data"
    label_dir="${task_dir}/label_data"
    label_vis_dir="${task_dir}/label_vis"
    data_dir="${task_dir}/data"

    mkdir -p "${endpose_dir}" "${label_dir}" "${label_vis_dir}"

    # 2. HDF5 -> JSON (Convert)
    if [[ -d "${data_dir}" ]]; then
        echo ">> Step 1: Converting HDF5 files..."
        for hdf5_file in "${data_dir}"/*.hdf5; do
            [[ -f "${hdf5_file}" ]] || continue
            basename_hdf5=$(basename "${hdf5_file}" .hdf5)
            json_path="${endpose_dir}/${basename_hdf5}.json"

            # Skip if json already exists
            if [[ ! -f "${json_path}" ]]; then
                python3 src/convert.py --hdf5_path "${hdf5_file}" --json_path "${json_path}"
            fi
        done
    else
        echo "Warning: No data folder in ${task_name}, skipping..."
        continue
    fi

    # 3. Resolve config (prefer per-task config, fall back to global override)
    # Convention: configs/config_<category>_<task>.yaml
    config_name="config_$(echo "${task_name}" | tr '/' '_').yaml"
    task_specific_config="./configs/${config_name}"

    if [[ -f "${task_specific_config}" && "${config_path_override}" == "${DEFAULT_CONFIG}" ]]; then
        current_config="${task_specific_config}"
    else
        current_config="${config_path_override}"
    fi
    echo ">> Using config: ${current_config}"

    # 4. Run labelling (rule-based atomic-action annotation)
    echo ">> Step 2: Running rule-based atomic-action annotation..."
    for json_file in "${endpose_dir}"/*.json; do
        [[ -f "${json_file}" ]] || continue
        basename_json=$(basename "${json_file}")
        output_file="${label_dir}/${basename_json}"

        python3 src/label.py \
            --config "${current_config}" \
            --input_file "${json_file}" \
            --output_file "${output_file}"
    done

    # 5. Visualize
    echo ">> Step 3: Visualizing results..."
    for json_file in "${label_dir}"/*.json; do
        [[ -f "${json_file}" ]] || continue
        basename_json=$(basename "${json_file}")
        basename_png="${basename_json%.json}.png"
        output_file="${label_vis_dir}/${basename_png}"

        python3 src/visualize_labels.py \
            --input "${json_file}" \
            --output "${output_file}"
    done

    echo ">> Task [ ${task_name} ] completed."
done

echo ""
echo "All tasks finished!"

# Example:
#   BASE_DATASET_DIR=/path/to/RoboTwin2.0/dataset bash batch_label.sh --all
