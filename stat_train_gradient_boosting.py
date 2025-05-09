import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# === User-configurable variables ===
# List any number of predictor column names from the Excel file
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
# Name of the target column to predict
TARGET = 'Severe_all'
# Excel file path
EXCEL_FILE = 'iProve_gen_70.xlsx'
# Gradient Boosting hyperparameters
N_ESTIMATORS = 100       # number of boosting stages
LEARNING_RATE = 0.1      # learning rate shrinks contribution of each tree
MAX_DEPTH = 3            # maximum depth of individual regression estimators
RANDOM_STATE = 42        # for reproducibility
# Train/test split
TEST_SIZE = 0.02          # fraction of data to reserve for validation
# Output model file
MODEL_OUTPUT = 'gb_model_severe.joblib'


def main():
    # Load raw data
    df = pd.read_excel(EXCEL_FILE)

    # Check required columns
    required = PREDICTORS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")
    print(f"All required columns present: {required}")

    # Subset and clean
    df = df[required].copy()
    df = df.dropna()
    if df.empty:
        raise ValueError("No data left after dropping rows with missing values.")
    # Check for infinite values
    arr = df.values
    if not np.isfinite(arr).all():
        raise ValueError("Data contains infinite values; please clean or impute before training.")

    # Split features and target
    X = df[PREDICTORS].values
    y = df[TARGET].values

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training on {len(y_train)} samples, validating on {len(y_val)} samples.")

    # Initialize and train model
    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate on validation set
    val_score = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")

    # Save model to disk
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Gradient Boosting model saved to '{MODEL_OUTPUT}'")


if __name__ == '__main__':
    main()
