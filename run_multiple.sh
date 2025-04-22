#!/bin/bash

CLASSES=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")
PROJECT_NAME="test-cifar-new-project"
BASE_RUN_NAME="1percent run"
BASE_EXTRA_IMAGES_DIR="/home/karol/sd-imbalance/cifar10_synaug_sdxl_class_specific"
EXTRA_IMAGES_PER_CLASS_VALUE=4950


NUM_PARALLEL_JOBS=$(nproc)


if ! command -v parallel &> /dev/null
then
    echo "Error: GNU Parallel is not installed."
    echo "Please install it using: sudo apt update && sudo apt install parallel"
    exit 1
fi

echo "Starting parallel training runs for ${#CLASSES[@]} classes..."
echo "Project: ${PROJECT_NAME}"
echo "Running a maximum of ${NUM_PARALLEL_JOBS} jobs concurrently."
echo "---"

run_training() {
  local class_name="$1"
  local run_name="${class_name}-${BASE_RUN_NAME}"
  local extra_images_dir="${BASE_EXTRA_IMAGES_DIR}/${class_name}"

  echo "Starting job for class: ${class_name}"
  echo "  Run Name: ${run_name}"
  echo "  Extra Images Dir: ${extra_images_dir}"

  nice -n 10 uv run train.py \
    name="${run_name}" \
    project="${PROJECT_NAME}" \
    add_extra_images=True \
    downsample_classes."${class_name}"="${DOWNSAMPLE_VALUE}" \
    extra_images_per_class."${class_name}"="${EXTRA_IMAGES_PER_CLASS_VALUE}" \
    extra_images_dir="${extra_images_dir}"

  local exit_status=$?
  if [ $exit_status -ne 0 ]; then
    echo "Warning: Job for class ${class_name} failed with exit status ${exit_status}."
  else
    echo "Finished job for class: ${class_name}"
  fi
  echo "---"
  return $exit_status
}

export -f run_training
export PROJECT_NAME BASE_RUN_NAME BASE_EXTRA_IMAGES_DIR DOWNSAMPLE_VALUE EXTRA_IMAGES_PER_CLASS_VALUE


printf "%s\n" "${CLASSES[@]}" | parallel \
  --jobs ${NUM_PARALLEL_JOBS} \
  --eta \
  --bar \
  --joblog parallel_cifar_jobs.log \
  run_training {}

echo "---"
echo "All parallel jobs launched. Check 'parallel_cifar_jobs.log' for execution details."
echo "Note: The script finishes when all background jobs managed by 'parallel' complete."