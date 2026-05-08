#!/bin/bash

# Defaults
config_path_override="config/config.yaml"
: "${BASE_DATASET_DIR:=./dataset}"

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

task_dir="${BASE_DATASET_DIR}/${task_name}"

endpose_dir="${task_dir}/endpose_data"
label_dir="${task_dir}/label_data"
label_vis_dir="${task_dir}/label_vis"

# Ensure output dirs exist
for dir in "${endpose_dir}" "${label_dir}" "${label_vis_dir}"; do
    if [[ ! -d "${dir}" ]]; then
        mkdir -p "${dir}"
    fi
done

# Convert HDF5 -> JSON
data_dir="${task_dir}/data"
if [[ -d "${data_dir}" ]]; then
    for hdf5_file in "${data_dir}"/*.hdf5; do
        if [[ ! -f "${hdf5_file}" ]]; then
            continue
        fi
        basename_hdf5=$(basename "${hdf5_file}" .hdf5)
        json_path="${endpose_dir}/${basename_hdf5}.json"
        python3 src/convert.py --hdf5_path "${hdf5_file}" --json_path "${json_path}"
    done
else
    echo "Warning: ${data_dir} does not exist."
fi

# Resolve config path
if [[ -n "${config_path_override}" ]]; then
    config_path="${config_path_override}"
    if [[ ! -f "${config_path}" ]]; then
        echo "Error: Provided config file '${config_path}' not found."
        exit 1
    fi
else
    config_name="config_$(echo "${task_name}" | tr '/' '_').yaml"
    config_path="./configs/${config_name}"
    if [[ ! -f "${config_path}" ]]; then
        echo "Error: Config file '${config_path}' not found."
        exit 1
    fi
fi

# Run label.py
for json_file in "${endpose_dir}"/*.json; do
    if [[ ! -f "${json_file}" ]]; then
        continue
    fi

    basename_json=$(basename "${json_file}")
    output_file="${label_dir}/${basename_json}"

    python3 src/label.py \
        --config "${config_path}" \
        --input_file "${json_file}" \
        --output_file "${output_file}"
done

# Run visualize_labels.py
for json_file in "${label_dir}"/*.json; do
    if [[ ! -f "${json_file}" ]]; then
        continue
    fi

    basename_json=$(basename "${json_file}")
    basename_png="${basename_json%.json}.png"
    output_file="${label_vis_dir}/${basename_png}"

    python3 src/visualize_labels.py \
        --input "${json_file}" \
        --output "${output_file}"
done

echo "All done for task: ${task_name}"
