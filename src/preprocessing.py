import os
import pandas as pd


def load_raw_data() -> pd.DataFrame:
    """Loads a CSV file from the data/raw directory."""
    df=pd.read_csv("data/raw/Raisin_Dataset.csv")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs cleaning and feature engineering operations."""
    print("Preprocessing data...")
    df_clean = df.copy()

    # Example cleaning step: drop missing values
    df_clean = df_clean.dropna()

    return df_clean


def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    """Saves the preprocessed DataFrame to the data/processed directory."""
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, file_name)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")