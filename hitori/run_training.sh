#!/bin/bash
# Hitori GRPO Training Script
# Run on 4xA100 cluster with wandb logging and HuggingFace Hub upload

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
export WANDB_PROJECT="hitori-grpo"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# Customize these
RUN_NAME="${1:-hitori-$(date +%Y%m%d-%H%M%S)}"
HF_MODEL_ID="${2:-}"  # Optional: HuggingFace Hub model ID (e.g., username/hitori-grpo)
OUTPUT_DIR="outputs/${RUN_NAME}"

echo "============================================================"
echo "Hitori GRPO Training"
echo "============================================================"
echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
if [ -n "${HF_MODEL_ID}" ]; then
    echo "HF Hub upload: ${HF_MODEL_ID}"
fi
echo "============================================================"

# =============================================================================
# Step 0: Check Authentication
# =============================================================================
# Check HuggingFace login (required for model download and upload)
if ! huggingface-cli whoami &>/dev/null; then
    echo "Not logged into HuggingFace. Please run: huggingface-cli login"
    exit 1
fi
echo "HuggingFace: logged in as $(huggingface-cli whoami | head -1)"

# Check Wandb login
if ! wandb verify &>/dev/null 2>&1; then
    echo "Warning: Wandb may not be configured. Run: wandb login"
fi

# =============================================================================
# Step 1: Download Model (if not already downloaded)
# =============================================================================
if [ ! -d "models/qwen2.5-3b-instruct" ]; then
    echo "Downloading model..."
    python download_model.py
else
    echo "Model already downloaded, skipping..."
fi

# =============================================================================
# Step 2: Generate Dataset (if not already generated)
# =============================================================================
if [ ! -d "data/hitoridata" ]; then
    echo "Generating dataset..."
    python dataset.py --train-examples 10000 --output-dir data/hitoridata
else
    echo "Dataset already exists, skipping..."
fi

# =============================================================================
# Step 3: Run Training
# =============================================================================
echo "Starting training..."

# Build training command
TRAIN_CMD="accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --num_processes 3 \
    train_hitori_grpo.py \
    --config configs/hitori_grpo.yaml \
    --output_dir ${OUTPUT_DIR} \
    --wandb_run_name ${RUN_NAME} \
    --report_to wandb"

# Add HuggingFace Hub upload if model ID is provided
if [ -n "${HF_MODEL_ID}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --hub_model_id ${HF_MODEL_ID}"
fi

# Run training
eval ${TRAIN_CMD}

echo "============================================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
if [ -n "${HF_MODEL_ID}" ]; then
    echo "Model uploaded to: https://huggingface.co/${HF_MODEL_ID}"
fi
echo "============================================================"
