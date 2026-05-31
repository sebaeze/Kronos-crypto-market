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
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume']]

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

    # Export the final clean dataset
    master_df.to_csv(output_csv_path, index=False)
    print(f"SUCCESS: Exported {len(master_df)} chronological rows to '{output_csv_path}'")

# Execute the pipeline
# Replace './my_json_folder' with the path to your actual folder containing the JSONs
build_kronos_dataset_from_folder(
    target_folder='C:\\00 - GITHUB\\volume-usdt-batch\\candles\\historical_futures\\GUAUSDT\\1m', 
    output_csv_path='kronos_combined_1m_training_data.csv'
)