import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def extract_data(file_name: str) -> pd.DataFrame:
    data_dir = "data"
    data_path = os.path.join(data_dir, file_name)
    abs_path = os.path.abspath(data_path)

    if not os.path.exists(abs_path):
        logging.error(f"❌ ERROR: '{file_name}' not found at {abs_path}")
        return pd.DataFrame()

    df = pd.read_csv(abs_path)
    logging.info(f"✅ Data loaded from {file_name}, shape: {df.shape}")
    return df
