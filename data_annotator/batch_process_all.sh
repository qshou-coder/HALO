#!/bin/bash

# Override via env var:
#   BASE_DATASET_DIR=/path/to/raw_data bash batch_process_all.sh
: "${BASE_DATASET_DIR:=./dataset}"

# Sanity check
if [[ ! -d "${BASE_DATASET_DIR}" ]]; then
    echo -e "\033[31mError: Base directory ${BASE_DATASET_DIR} does not exist.\033[0m"
    exit 1
fi

echo "Starting batch processing for all tasks in: ${BASE_DATASET_DIR}"

# Counters
processed_count=0
skipped_count=0

# Outer loop: category directories (e.g. lift_pot, adjust_bottle, ...)
for category_dir in "${BASE_DATASET_DIR}"/*; do
    if [[ ! -d "${category_dir}" ]]; then
        continue
    fi

    # Inner loop: concrete task directories (e.g. aloha-agilex_clean_50)
    for task_dir in "${category_dir}"/*; do
        if [[ ! -d "${task_dir}" ]]; then
            continue
        fi

        # Force-refresh HDFS-style mounts so the existence check below is reliable
        ls "${task_dir}" > /dev/null 2>&1

        # Skip if already processed (complete_annotation_data folder exists)
        if [[ -d "${task_dir}/complete_annotation_data" ]]; then
            echo -e "\033[33m[SKIP]\033[0m Task already processed: $(basename "${task_dir}")"
            ((skipped_count++))
            continue
        fi

        # Make sure this is actually a task folder (has label_data inside)
        if [[ ! -d "${task_dir}/label_data" ]]; then
            echo "Skipping ${task_dir}: 'label_data' directory not found."
            continue
        fi

        task_name=$(basename "${category_dir}")/$(basename "${task_dir}")
        echo "=========================================================="
        echo -e "\033[34mProcessing Task:\033[0m ${task_name}"
        echo "Path: ${task_dir}"
        echo "=========================================================="

        # Delegate to the per-task pipeline
        bash process_annotation_complete.sh --task_name "${task_dir}"

        if [[ $? -eq 0 ]]; then
            echo -e "\033[32mSuccessfully finished task:\033[0m ${task_name}"
            ((processed_count++))
        else
            echo -e "\033[31mError occurred while processing task:\033[0m ${task_name}"
            # Stop on first failure; comment out the next line to continue instead.
            exit 1
        fi
    done
done

echo "----------------------------------------------------------"
echo -e "\033[32mBatch processing complete!\033[0m"
echo "Total processed: ${processed_count}"
echo "Total skipped: ${skipped_count}"
echo "----------------------------------------------------------"
