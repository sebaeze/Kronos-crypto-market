import os
import json
import glob
import pandas as pd
import numpy as np

def load_from_json_folder(folder_path):
    """
    Reads all *.json files in the given folder_path.
    Each file is expected to contain a JSON array of Binance-style kline objects.
    Combines them into a single sorted pandas DataFrame.
    """
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return pd.DataFrame()

    all_data = []
    for file_path in json_files:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                all_data.extend(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Rename and convert columns
    if "openTime" in df.columns:
        df["timestamps"] = pd.to_datetime(df["openTime"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms" if str(df["timestamp"].iloc[0]).isdigit() and int(df["timestamp"].iloc[0]) > 1e11 else "s")

    # Map amount column from quoteAssetVolume
    if "quoteAssetVolume" in df.columns:
        df["amount"] = df["quoteAssetVolume"]
    elif "takerBuyQuoteAssetVolume" in df.columns and "amount" not in df.columns:
        df["amount"] = df["takerBuyQuoteAssetVolume"]

    return preprocess_ohlcv(df)

def load_from_csv(file_path):
    """
    Loads OHLCV data from a standard CSV file.
    """
    if not os.path.exists(file_path):
        print(f"CSV file not found: {file_path}")
        return pd.DataFrame()

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    if "timestamps" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "timestamps"}, inplace=True)
    elif "timestamps" not in df.columns and "date" in df.columns:
        df.rename(columns={"date": "timestamps"}, inplace=True)

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return preprocess_ohlcv(df)

def preprocess_ohlcv(df):
    """
    Cleans, sorts, and checks for missing values.
    Computes 'amount' if missing.
    Returns standard columns: ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
    """
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "amount" in df.columns:
         df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Sort by timestamp
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Forward fill missing values
    df = df.ffill()
    # Backward fill any remaining NaNs at the beginning
    df = df.bfill()

    price_cols = ["open", "high", "low", "close"]
    # Check if we have all necessary columns
    for col in price_cols + ["volume"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the data.")

    # Calculate amount if missing
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[price_cols].mean(axis=1)

    # Return standard columns
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].copy()

def chunk_data(df, sequence_length, step=None):
    """
    Splits a long sequence of data into contiguous sequences of a specific size.
    Returns a list of DataFrames, each of size `sequence_length`.
    """
    if step is None:
        step = sequence_length

    chunks = []
    total_len = len(df)
    
    for i in range(0, total_len - sequence_length + 1, step):
        chunk = df.iloc[i : i + sequence_length].copy().reset_index(drop=True)
        chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    # Self-test using the finetune_ohlcv_json directory
    sample_dir = "finetune_ohlcv_json"
    if os.path.exists(sample_dir):
        print("Testing JSON loader...")
        df = load_from_json_folder(sample_dir)
        print(f"Loaded {len(df)} rows.")
        if len(df) > 0:
            print(df.head())
            
            # Test chunking
            chunks = chunk_data(df, sequence_length=561)
            print(f"Created {len(chunks)} chunks of length 561.")
    else:
        print(f"Sample directory {sample_dir} not found. Skip testing.")
