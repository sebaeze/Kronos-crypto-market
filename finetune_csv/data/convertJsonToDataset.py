import pandas as pd
import json
from pathlib import Path

def process_single_json(file_path):
    """Parses a single JSON file and returns a formatted DataFrame."""
    with open(file_path, 'r') as file:
        raw_data = json.load(file)

    df = pd.DataFrame(raw_data)

    # 1. Convert string prices and volumes to floats
    target_cols = ['open', 'high', 'low', 'close', 'volume']
    df[target_cols] = df[target_cols].astype(float)

    # 2. Convert Unix milliseconds to readable datetime format
    df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')

    # 3. Filter down to strictly what KronosPredictor requires
    result = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()

    # 4. Zero out volume field
    ## result['volume'] = 0

    return result

def split_on_structural_breaks(df, price_jump_threshold=0.10, min_segment_len=512):
    """
    Splits a DataFrame into valid continuous sub-segments based on the 
    Price Jump Threshold (Theta_jump) and discards short segments.
    """
    df = df.sort_values(by='timestamps').reset_index(drop=True)
    
    # Calculate price jump: |open_t / close_{t-1} - 1|
    prev_close = df['close'].shift(1)
    relative_jump = (df['open'] / prev_close - 1).abs()
    
    # Flag rows where a jump occurs (excluding the first row)
    jump_mask = (relative_jump > price_jump_threshold) & (df.index > 0)
    
    # Assign group IDs based on cumulative sum of breaks
    df['segment_id'] = jump_mask.cumsum()
    
    # Filter out segments that do not meet the minimum length
    valid_segments = []
    for seg_id, group in df.groupby('segment_id'):
        if len(group) >= min_segment_len:
            valid_segments.append(group.drop(columns=['segment_id']))
            
    if not valid_segments:
        return pd.DataFrame(columns=[c for c in df.columns if c != 'segment_id'])
        
    return pd.concat(valid_segments, ignore_index=True)

def build_kronos_dataset_from_folder(target_folder, output_csv_path):
    """Iterates through a directory of JSONs to build a unified time-series dataset."""
    folder_path = Path(target_folder)
    
    # Grab all .json files in the target directory
    json_files = list(folder_path.glob('*.json'))
    
    if not json_files:
        print(f"Error: No JSON files found in directory '{target_folder}'")
        return

    print(f"Found {len(json_files)} JSON files. Starting batch processing...")
    
    dataframes = []
    
    # Process each file individually
    for json_file in json_files:
        try:
            df = process_single_json(json_file)
            dataframes.append(df)
            print(f" [+] Processed: {json_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f" [-] Failed to process {json_file.name}: {e}")

    if not dataframes:
        print("Error: No valid data could be extracted from the files.")
        return

    # Combine all individual DataFrames into one master DataFrame
    print("\nMerging data...")
    master_df = pd.concat(dataframes, ignore_index=True)

    # CRITICAL: Sort chronologically to ensure the time-series sequence is intact
    master_df = master_df.sort_values(by='timestamps').reset_index(drop=True)

    # CRITICAL: Drop overlapping candles that might exist between file exports
    initial_row_count = len(master_df)
    master_df = master_df.drop_duplicates(subset=['timestamps'], keep='last')
    duplicates_removed = initial_row_count - len(master_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate timestamps due to overlapping files.")

    # CRITICAL: Filter out segments with structural breaks > 10%
    print("Applying structural break segmentation filter...")
    initial_len = len(master_df)
    master_df = split_on_structural_breaks(master_df, price_jump_threshold=0.10, min_segment_len=512)
    filtered_out = initial_len - len(master_df)
    if filtered_out > 0:
        print(f"Filtered out {filtered_out} rows due to structural breaks or insufficient segment length.")

    # Export the final clean dataset
    master_df.to_csv(output_csv_path, index=False)
    print(f"SUCCESS: Exported {len(master_df)} chronological rows to '{output_csv_path}'")

# Execute the pipeline
# Replace './my_json_folder' with the path to your actual folder containing the JSONs
build_kronos_dataset_from_folder(
    target_folder='C:\\00 - GITHUB\\volume-usdt-batch\\candles\\historical_futures\\BNBUSDT\\1m', 
    output_csv_path='kronos_combined_1m_training_data.csv'
)