# Kronos CPU Fine-Tuning Guide

This guide walks you through setting up your local environment and running the fine-tuning pipeline for the `NeoQuasar/Kronos-base` model entirely on your CPU.

## Prerequisites

- Windows PowerShell
- Git
- Python 3.9+
- At least 32GB of system RAM

## Step 1: Environment Setup

We have provided a script that automates the creation of an isolated virtual environment, installs the CPU-only version of PyTorch, and installs all required dependencies.

Open a PowerShell terminal in the repository root and run:

```powershell
.\setup_env.ps1 -Symbol "JELLYJELLYUSDT"
```

> **Note:** The `-Symbol` parameter defines the crypto asset you are training on. This creates a dedicated git branch (`feature/local-finetuning_JELLYJELLYUSDT`) and configures your model export paths automatically.

## Step 2: Preparing Data

Ensure your historical OHLCV data is exported as Binance-style JSON klines. 
Place all `*.json` files inside the `finetune_ohlcv_json/` directory in the repository root. The training script will automatically locate, merge, sort, and chunk these files for you.

## Step 3: Run Training

You do not need to manually activate the virtual environment if you use the provided runner script. To execute training with the default configuration, run:

```powershell
.\run_training.ps1
```

### Customizing the Run

You can adjust hyperparameters directly via PowerShell arguments:

```powershell
.\run_training.ps1 -Symbol "JELLYJELLYUSDT" -Epochs 5 -BatchSize 8 -AccumSteps 2
```

The script manages memory by utilizing a small batch size, gradient accumulation, and restricting PyTorch to 4 threads to prevent operating system lockups.

## Step 4: Verify Local Inference

Once training completes, the fine-tuned model and tokenizer will be saved locally to:
- `./kronos-finetuned-{Symbol}/model/`
- `./kronos-finetuned-{Symbol}/tokenizer/`

*(These massive weight files are safely ignored by `.gitignore` so you don't accidentally push them).*

To run a quick local inference check and verify the model outputs standard data, execute:

```powershell
.\.venv\Scripts\python.exe local_inference.py --symbol "JELLYJELLYUSDT"
```

## Step 5: Save Code to Git

To push your training scripts and pipeline (without the heavy model weights) to your GitHub repository, run the helper script:

```powershell
.\git_push.ps1 -Symbol "JELLYJELLYUSDT"
```
