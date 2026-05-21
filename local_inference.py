import os
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from model.kronos import Kronos, KronosTokenizer, KronosPredictor

def create_dummy_data(sequence_length=512):
    """Creates a dummy OHLCV DataFrame for testing."""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(sequence_length)][::-1]
    
    # Generate random walk prices
    prices = np.cumsum(np.random.randn(sequence_length) * 0.5) + 100
    
    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": prices + np.random.rand(sequence_length) * 0.2,
        "high": prices + np.random.rand(sequence_length) * 0.5,
        "low": prices - np.random.rand(sequence_length) * 0.5,
        "close": prices + np.random.rand(sequence_length) * 0.2,
        "volume": np.random.randint(100, 10000, size=sequence_length)
    })
    
    # Calculate amount
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default="JELLYJELLYUSDT", help="Crypto symbol for naming outputs")
    parser.add_argument('--pred-len', type=int, default=48, help="Number of future steps to predict")
    args = parser.parse_args()

    model_dir = f"./kronos-finetuned-{args.symbol}/model"
    tokenizer_dir = f"./kronos-finetuned-{args.symbol}/tokenizer"
    
    if not os.path.exists(model_dir) or not os.path.exists(tokenizer_dir):
        print(f"Error: Fine-tuned model or tokenizer not found for symbol {args.symbol}.")
        print(f"Please ensure you've run train_kronos.py first and that {model_dir} and {tokenizer_dir} exist.")
        return

    device = "cpu"
    print(f"Loading local fine-tuned model and tokenizer from {model_dir} on {device}...")
    
    # Load the local model and tokenizer
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device)
    model = Kronos.from_pretrained(model_dir).to(device)
    
    tokenizer.eval()
    model.eval()

    # Initialize the predictor
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # Create dummy data
    print("Generating dummy historical data for inference...")
    context_df = create_dummy_data(sequence_length=256)
    
    # Prepare timestamps
    x_timestamp = context_df["timestamps"].reset_index(drop=True)
    last_timestamp = x_timestamp.iloc[-1]
    
    # Create future timestamps
    future_timestamps = [last_timestamp + timedelta(minutes=i+1) for i in range(args.pred_len)]
    y_timestamp = pd.Series(future_timestamps)

    # Run inference
    print(f"Running prediction for {args.pred_len} steps...")
    with torch.no_grad():
        pred_df = predictor.predict(
            df=context_df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True),
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=args.pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
        )

    print("\n--- Inference Complete ---")
    print("Predicted DataFrame (first 5 rows):")
    print(pred_df.head())
    
    print("\nPredicted DataFrame (last 5 rows):")
    print(pred_df.tail())

if __name__ == "__main__":
    main()
