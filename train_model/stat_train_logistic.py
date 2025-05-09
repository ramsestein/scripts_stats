import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# === User-configurable variables ===
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
TARGET = 'PPC_all'                            # target column name
EXCEL_FILE = 'iProve_gen_70.xlsx'
# Logistic Regression hyperparameters
PENALTY = 'l2'          # norm used in the penalization: 'l1', 'l2', 'elasticnet', or 'none'
C = 2.0                 # inverse of regularization strength; smaller values specify stronger regularization
SOLVER = 'lbfgs'        # algorithm to use in the optimization: 'liblinear', 'lbfgs', 'saga', etc.
RANDOM_STATE = 42       # for reproducibility
MAX_ITER = 100          # maximum number of iterations for convergence
# Train/test split fraction
TEST_SIZE = 0.02         # fraction of data reserved for validation
# Output model file
MODEL_OUTPUT = 'logistic_model.joblib'


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

    # Initialize and train Logistic Regression
    model = LogisticRegression(
        penalty=PENALTY,
        C=C,
        solver=SOLVER,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Validation accuracy
    val_acc = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Save the model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Logistic Regression model saved to '{MODEL_OUTPUT}'")


if __name__ == '__main__':
    main()
