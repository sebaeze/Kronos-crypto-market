## Step-by-step

## Steps

```bash
git clone https://github.com/sebaeze/Kronos-crypto-market
cd Kronos-crypto-market
pip install -r requirements.txt

export PYTHONPATH=$PWD
pip install comet_ml



torchrun --nproc_per_node=1 finetune/train_tokenizer.py

torchrun --nproc_per_node=1 finetune/train_predictor.py



zip -r my_trained_kronos_v2.zip ./output_models
gsutil cp my_trained_kronos_v2.zip gs://kronos-1m-workspace/
```