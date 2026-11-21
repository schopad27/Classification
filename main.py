from src.data_preparation import prepare_data
from src.eda import LABEL_COLUMN, run_eda
from src.get_data import run_get_data
from src.optimize_model import predict_interactive, run_optimize
from src.shortlist import run_shortlist

RAW_DATA_PATH = "data/raw/Raisin_Dataset.csv"


def run_training():
    raw_df = run_get_data(RAW_DATA_PATH)
    print("\nGet Data step completed.")

    train_df, test_df = run_eda(raw_df)
    print("\nEDA step completed.")

    prepared = prepare_data(train_df, test_df)
    print("\nData Preparation step completed.")
    if prepared is None:
        model_train, model_test = train_df, test_df
    else:
        model_train, model_test = prepared

    shortlist_summary = run_shortlist(
        model_train,
        model_test,
        label_column=LABEL_COLUMN,
        output_dir="outputs/model_shortlist",
        primary_metric="roc_auc",
    )
    print("\nShortlist step completed.")

    run_optimize(
        model_train,
        model_test,
        label_column=LABEL_COLUMN,
        shortlist=shortlist_summary["shortlist"],
        output_dir="outputs/optimize_model",
    )
    print("\nOptimize step completed.")


def run_prediction():
    predict_interactive()


def main():
    print("1. Train (Get Data through fine-tune and save the best model)")
    print("Anything else: Predict with the saved best model")
    choice = input("Enter choice: ").strip()
    match choice:
        case "1":
            run_training()
        case _:
            run_prediction()


if __name__ == "__main__":
    main()
