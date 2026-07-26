from src.preprocessing import load_raw_data, preprocess_data, save_processed_data


def main():
    # 1. Define filenames
    #raw_filename = "dataset.csv"  # Name of your file in data/raw/
    processed_filename = "cleaned_raisin_dataset.csv"

    # 2. Load
    raw_df = load_raw_data()

    # 3. Process
    cleaned_df = preprocess_data(raw_df)

    # 4. Save
    save_processed_data(cleaned_df, processed_filename)

    print("\nPipeline executed successfully!")
    print(cleaned_df.head())


if __name__ == "__main__":
    main()