import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from finetune.config import Config

class QlibDataset(Dataset):
    """
    A custom PyTorch Dataset that reads raw CSV time-series data, 
    pads the price dimensions, and extracts time embeddings for the Predictor.
    """
    def __init__(self, split='train'):
        self.config = Config()
        self.split = split
        
        # 1. Read the CSV file
        df = pd.read_csv(self.config.dataset_path)
        
        # 2. Process Timestamps and extract Time Features
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            # Extract the specific time features Kronos requires
            df['minute'] = df['timestamps'].dt.minute
            df['hour'] = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day'] = df['timestamps'].dt.day
            df['month'] = df['timestamps'].dt.month
        else:
            raise ValueError("CSV must contain a 'timestamps' column.")
        
        # 3. Separate Price Data and Time Data
        data_values = df[self.config.feature_list].values
        time_values = df[self.config.time_feature_list].values
        
        # 4. Create the 80/20 Train/Validation Split
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
        
        # --- PRICE DATA PROCESSING ---
        window = self.data[start_idx:end_idx]
        x_tensor = torch.tensor(window, dtype=torch.float32)
        
        # Pad the 5 features (OHLCV) to 6 features to match Kronos architecture
        pad_column = torch.zeros((self.seq_len, 1), dtype=torch.float32)
        padded_x_tensor = torch.cat([x_tensor, pad_column], dim=1) 
        
        # --- TIME DATA PROCESSING ---
        window_time = self.time_data[start_idx:end_idx]
        # Time embeddings require integer types (long), not floats
        time_tensor = torch.tensor(window_time, dtype=torch.long)
        
        # Return both the padded price matrix and the time feature matrix
        return (padded_x_tensor, time_tensor)