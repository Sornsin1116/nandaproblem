import os
import pandas as pd
import json

def load_data(dfs_dict, path: str):
    os.makedirs(path, exist_ok=True)
    data_dict, mapping = dfs_dict

    # Convert numpy.int64 to int
    mapping = {k: int(v) for k, v in mapping.items()}

    # Save DataFrames
    for key, df in data_dict.items():
        file_path = os.path.join(path, f"{key}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {key} to {file_path}")

    # Save mapping
    mapping_path = os.path.join(path, "disease_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    print("Saved disease mapping to", mapping_path)
