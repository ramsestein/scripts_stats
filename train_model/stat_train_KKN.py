import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# === User-configurable variables ===
# List of predictor column names in the Excel file
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
# Name of the target column to predict
TARGET = 'Severe_all'
# Path to Excel file
EXCEL_FILE = 'iProve_gen_70.xlsx'
# k-NN hyperparameters
N_NEIGHBORS = 5           # number of neighbors to use
WEIGHTS = 'uniform'       # weight function: 'uniform' or 'distance'
METRIC = 'minkowski'      # distance metric to use: 'euclidean', 'manhattan', 'minkowski', etc.
P = 2                     # power parameter for Minkowski metric (p=2 is Euclidean)
# Train/test split fraction
TEST_SIZE = 0.02           # fraction of data reserved for validation
RANDOM_STATE = 42         # for reproducibility of train/test split
# Output model file
MODEL_OUTPUT = 'knn_model_severe.joblib'


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
    arr = df.values
    if not np.isfinite(arr).all():
        raise ValueError("Data contains infinite values; clean or impute before training.")

    # Features and target
    X = df[PREDICTORS].values
    y = df[TARGET].values

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training on {len(y_train)} samples, validating on {len(y_val)} samples.")

    # Initialize and train k-NN
    model = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights=WEIGHTS,
        metric=METRIC,
        p=P
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Validation accuracy
    val_acc = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Save the model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"k-NN model saved to '{MODEL_OUTPUT}'")


if __name__ == '__main__':
    main()
