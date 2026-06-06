#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# BUCKET_NAME is passed as an environment variable from the PowerShell command
echo "Downloading dataset from GCS..."
mkdir -p ./finetune_csv/data/
gsutil cp gs://$BUCKET_NAME/kronos_combined_1m_training_data.csv ./finetune_csv/data/

echo "Starting Stage 1: Tokenizer Training..."
torchrun --nproc_per_node=1 finetune/train_tokenizer.py

echo "Starting Stage 2: Predictor Training..."
torchrun --nproc_per_node=1 finetune/train_predictor.py

echo "Zipping output models..."
zip -r trained_model_opnusdt.zip ./output_models

echo "Uploading trained model to GCS..."
# Appending a timestamp so you don't overwrite previous runs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
gsutil cp trained_model_opnusdt.zip gs://$BUCKET_NAME/models/kronos_opnusdt_$TIMESTAMP.zip

echo "Pipeline execution complete. Shutting down."