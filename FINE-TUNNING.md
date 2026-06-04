# Kronos 1-Minute Fine-Tuning Guide (GCP Vertex AI)

This guide provides a comprehensive, beginner-friendly walkthrough for setting up cloud compute on Google Cloud Platform (GCP) and fine-tuning the [Kronos](https://github.com/shiyu-coder/Kronos) time-series foundation model on custom 1-minute cryptocurrency data (e.g., GUA/USDT).

By bypassing the complex Microsoft Qlib requirement, this setup allows you to train directly on raw Binance-style JSON/CSV exports using a single, cost-effective NVIDIA L4 GPU.

---

## Phase 1: Setting up Google Cloud Platform (GCP)

1. **Create a Project:** Go to the [Google Cloud Console](https://console.cloud.google.com/), click the project dropdown at the top left, and create a **New Project** (e.g., `Kronos-Training`).
2. **Enable Billing:** Ensure billing is linked to your project. (New accounts typically receive a $300 free trial, which covers this workflow).
3. **Enable Vertex AI:** Search for "Vertex AI" in the top search bar and click **Enable All Recommended APIs**.

---

## Phase 2: Renting the Cloud GPU (Vertex AI Workbench)

We will use Vertex AI Workbench to get a visual IDE (JupyterLab) connected directly to a cloud GPU.

1. Navigate to **Vertex AI > Workbench** in the left menu.
2. Under the **Instances** tab, click **+ Create New**.
3. **Configuration:**
   * **Name:** `kronos-gpu-workspace`
   * **Region:** `us-central1` (Most cost-effective for GPUs).
   * **Machine Setup > GPU:** Select **NVIDIA L4** (Count: 1). The Machine Type will automatically adjust to `g2-standard-4`.
4. Click **Create** and wait 3-5 minutes for the instance to provision.
5. Click **Open JupyterLab** next to your instance to launch the cloud IDE.

---

## Phase 3: Environment & Repository Setup

Inside your JupyterLab workspace:

1. Open a **Terminal** window (File > New > Terminal).
2. Clone your repository:
   ```bash
   git clone [https://github.com/sebaeze/Kronos-crypto-market](https://github.com/sebaeze/Kronos-crypto-market)
   cd Kronos-crypto-market
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install comet_ml
   ```
4. Set your Python path so the training scripts can find the model architecture:
   ```bash
   export PYTHONPATH=$PWD
   ```
5. **Upload your Data:** In the left-hand file browser, navigate into `Kronos-crypto-market`. Create a folder named `raw_data` and drag-and-drop your 1-minute JSON files from your computer into this folder.

---

## Phase 4: Data Preparation

We need to merge the raw JSONs, sort them chronologically, drop overlapping duplicates, and export them as a clean CSV.

1. Create a file named `prepare_data.py` in the main `Kronos-crypto-market` directory.
2. Paste the following script:

```python
import pandas as pd
import json
from pathlib import Path
import os

def build_kronos_dataset_from_folder(target_folder, output_csv_path):
    folder_path = Path(target_folder)
    json_files = list(folder_path.glob('*.json'))
    
    dataframes = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            raw_data = json.load(file)
        df = pd.DataFrame(raw_data)
        target_cols = ['open', 'high', 'low', 'close', 'volume']
        df[target_cols] = df[target_cols].astype(float)
        df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')
        dataframes.append(df[['timestamps', 'open', 'high', 'low', 'close', 'volume']])

    print("Merging and sorting data...")
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # CRITICAL: Ensure chronological order and remove overlaps
    master_df = master_df.sort_values(by='timestamps').reset_index(drop=True)
    master_df = master_df.drop_duplicates(subset=['timestamps'], keep='last')
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    master_df.to_csv(output_csv_path, index=False)
    print(f"SUCCESS: Exported {len(master_df)} rows to {output_csv_path}")

build_kronos_dataset_from_folder('./raw_data', './dataset/opnusdt_1m.csv')
```
3. Run the script in your terminal:
   ```bash
   python prepare_data.py
   ```

---

## Phase 5: Model Configuration

Update the configuration file to handle our custom paths, batch sizes for the L4 GPU, and necessary optimizer variables.

1. Open `finetune/config.py`.
2. Replace its contents with:

```python
class Config:
    def __init__(self):
        # --- Core Training Variables ---
        self.seed = 42
        self.accumulation_steps = 1   
        self.log_steps = 10           
        self.log_interval = 10        
        self.n_train_iter = 1000      
        self.n_valid_iter = 200       
        
        # --- Paths and Instrument ---
        self.dataset_path = "./dataset/opnusdt_1m.csv"
        self.instrument = 'opnusdt'
        self.dataset_begin_time = "2020-01-01"
        self.dataset_end_time = '2030-01-01'

        # --- Model Architecture ---
        self.lookback_window = 400   
        self.predict_window = 60    
        self.max_context = 512       
        self.feature_list = ['open', 'high', 'low', 'close', 'volume']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # --- Hardware & Optimizers ---
        self.device = 'cuda'
        self.epochs = 3               # Keep low to prevent overfitting on 1m data
        self.batch_size = 8
        self.tokenizer_learning_rate = 1e-5   
        self.predictor_learning_rate = 1e-5   
        self.adam_weight_decay = 1e-4
        self.adam_beta1 = 0.9           
        self.adam_beta2 = 0.999         
        
        # --- Save Paths ---
        self.save_path = "./output_models"
        self.tokenizer_save_folder_name = "tokenizer_finetuned"
        self.predictor_save_folder_name = "predictor_finetuned"
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # --- Extras ---
        self.use_comet = False      
        self.backtest_benchmark = "opnusdt"
```

---

## Phase 6: Custom Data Loader (The Qlib Bypass)

Kronos natively requires Microsoft Qlib and a 6-feature dataset. We must replace the data loader to read our CSV directly, pad our 5 features (OHLCV) to 6 features (with zeros), and dynamically extract Time Embeddings.

1. Open `finetune/dataset.py`.
2. Replace its contents with:

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from finetune.config import Config

class QlibDataset(Dataset):
    def __init__(self, split='train'):
        self.config = Config()
        self.split = split
        
        df = pd.read_csv(self.config.dataset_path)
        
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            # Extract time features required for Predictor embeddings
            df['minute'] = df['timestamps'].dt.minute
            df['hour'] = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day'] = df['timestamps'].dt.day
            df['month'] = df['timestamps'].dt.month
        else:
            raise ValueError("CSV must contain a 'timestamps' column.")
        
        data_values = df[self.config.feature_list].values
        time_values = df[self.config.time_feature_list].values
        
        # 80/20 Split
        split_idx = int(len(data_values) * 0.8)
        
        if split == 'train':
            self.data = data_values[:split_idx]
            self.time_data = time_values[:split_idx]
            self.n_samples = self.config.n_train_iter * self.config.batch_size
        else:
            self.data = data_values[split_idx:]
            self.time_data = time_values[split_idx:]
            self.n_samples = self.config.n_valid_iter * self.config.batch_size

        self.seq_len = self.config.max_context
        if len(self.data) < self.seq_len:
            raise ValueError(f"Not enough data in {split} split.")

    def __len__(self):
        return self.n_samples

    def set_epoch_seed(self, seed):
        np.random.seed(seed)

    def __getitem__(self, idx):
        max_start = len(self.data) - self.seq_len - 1
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + self.seq_len
        
        # --- PRICE DATA (Matrix Padding 5 -> 6) ---
        window = self.data[start_idx:end_idx]
        x_tensor = torch.tensor(window, dtype=torch.float32)
        pad_column = torch.zeros((self.seq_len, 1), dtype=torch.float32)
        padded_x_tensor = torch.cat([x_tensor, pad_column], dim=1) 
        
        # --- TIME DATA ---
        window_time = self.time_data[start_idx:end_idx]
        time_tensor = torch.tensor(window_time, dtype=torch.long)
        
        return (padded_x_tensor, time_tensor)
```

---

## Phase 7: Training the Model

Kronos utilizes a two-stage training process and requires the PyTorch distributed launcher (`torchrun`).

1. **Stage 1: Train the Tokenizer**
   ```bash
   torchrun --nproc_per_node=1 finetune/train_tokenizer.py
   ```
   *(Wait for this to complete. It adapts the tokenizer to the specific volatility of your 1m data).*

2. **Stage 2: Train the Predictor**
   ```bash
   torchrun --nproc_per_node=1 finetune/train_predictor.py
   ```
   *(This trains the main 102M parameter transformer model).*

---

## Phase 8: Secure Model Export

Downloading large `.safetensors` files directly through the JupyterLab browser can cause silent file corruption (e.g., `Error while deserializing header: incomplete metadata`). Use Google Cloud Storage for a secure transfer.

1. Zip your trained models:
   ```bash
   zip -r my_trained_kronos.zip ./output_models
   ```
2. Create a storage bucket in your GCP console (e.g., `my-kronos-bucket`).
3. Upload the zip directly via terminal:
   ```bash
   gsutil cp my_trained_kronos.zip gs://my-kronos-bucket/
   ```
4. Navigate to **Cloud Storage** in your Google Cloud Console and download the `.zip` file to your local machine.

---

## Phase 9: Clean Up (Avoid Charges)

**CRITICAL:** When you are done training, you must turn off the GPU machine to stop hourly billing.
1. Return to the GCP **Vertex AI > Workbench** dashboard.
2. Select your instance (`kronos-gpu-workspace`).
3. Click the **STOP** button at the top of the screen (or **DELETE** if you no longer need the environment).