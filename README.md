# Classification

Practicing two-class classification with a modular machine learning project structure.

## Project Structure

```text
Classification/
├── data/               # Dataset storage
│   ├── raw/            # Original, immutable data
│   └── processed/      # Cleaned and transformed data
├── notebooks/          # Jupyter notebooks for exploration and prototyping
├── src/                # Source code for the project
│   ├── __init__.py
│   ├── data_loader.py  # Scripts to load and split data
│   ├── preprocessing.py # Data cleaning and feature engineering
│   ├── model.py        # Model architecture definition
│   ├── train.py        # Training loop and logic
│   └── evaluate.py     # Evaluation metrics and visualization
├── config/             # Configuration files (YAML/JSON)
│   └── config.yaml
├── models/             # Saved model checkpoints and weights
├── requirements.txt    # Project dependencies
├── main.py             # Entry point to run the pipeline
└── README.md           # Project documentation
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure parameters in `config/config.yaml`.
3. Run the pipeline:
   ```bash
   python main.py
   ```
