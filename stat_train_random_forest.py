import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# === User-configurable variables ===
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
TARGET = 'Severe_all'                            # target column name
EXCEL_FILE = 'iProve_gen_70.xlsx'
# Random Forest hyperparameters
N_ESTIMATORS = 100      # number of trees in the forest
MAX_DEPTH = None        # maximum depth of each tree (None = nodes expanded until all leaves are pure)
MIN_SAMPLES_SPLIT = 2   # minimum samples required to split an internal node
MIN_SAMPLES_LEAF = 1    # minimum samples required to be at a leaf node
RANDOM_STATE = 42       # for reproducibility
# Train/test split fraction
TEST_SIZE = 0.02         # fraction of data reserved for validation
# Output model file
MODEL_OUTPUT = 'rf_model_severe.joblib'


def main():
    # Load data
    df = pd.read_excel(EXCEL_FILE)

    # Verify required columns
    required = PREDICTORS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel file: {missing}")
    print(f"All required columns present: {required}")

    # Subset and clean
    df = df[required].copy()
    df = df.dropna()
    if df.empty:
        raise ValueError("No data left after dropping rows with missing values.")
    # Check for infinite values
    if not np.isfinite(df.values).all():
        raise ValueError("Data contains infinite values; clean or impute before training.")

    # Features and target
    X = df[PREDICTORS].values
    y = df[TARGET].values

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training on {len(y_train)} samples, validating on {len(y_val)} samples.")

    # Initialize and train Random Forest
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Validation accuracy
    val_acc = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Save the model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Random Forest model saved to '{MODEL_OUTPUT}'")

if __name__ == '__main__':
    main()
