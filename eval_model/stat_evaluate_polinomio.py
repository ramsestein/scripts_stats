import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# === User-configurable variables ===
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
TARGET = 'PPC_all'                            # target column name
EXCEL_FILE = 'iProve_gen_30.xlsx'
BEST_DEGREE = 3                                     # degree of saved best model
MODEL_FILE = f'polynomial_model_deg.joblib'  # saved pipeline file
THRESHOLD = None  # not used; regression model

# Output files
PREDICTIONS_FILE = 'poly_predictions.xlsx'
METRICS_FILE = 'poly_metrics.xlsx'


def main():
    # Load data
    df = pd.read_excel(EXCEL_FILE)
    required = PREDICTORS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"All required columns present: {required}")

    # Clean data
    df = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        print("No valid rows to evaluate.")
        return

    # Load model
    pipeline = joblib.load(MODEL_FILE)
    print(f"Loaded polynomial pipeline for degree {BEST_DEGREE} from '{MODEL_FILE}'")

    # Prepare lists
    indices = []
    true_vals = []
    pred_vals = []

    # Row-by-row predictions
    for idx, row in df.iterrows():
        # Extract predictors
        X_row = row[PREDICTORS].values.reshape(1, -1)
        y_true = row[TARGET]
        y_pred = pipeline.predict(X_row)[0]
        # Print per-row
        preds_dict = {col: row[col] for col in PREDICTORS}
        print(f"Row {idx}: Predictors={preds_dict}, True={y_true}, Predicted={y_pred:.4f}")
        # Store
        indices.append(idx)
        true_vals.append(y_true)
        pred_vals.append(y_pred)

    # Convert to arrays
    y_true = np.array(true_vals)
    y_pred = np.array(pred_vals)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

    # Print overall metrics
    print("\nOverall Performance Metrics:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    # Save predictions
    out_df = df.loc[indices, PREDICTORS + [TARGET]].copy()
    out_df['Predicted'] = y_pred
    out_df.to_excel(PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to '{PREDICTIONS_FILE}'")

    # Save metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_excel(METRICS_FILE, index=False)
    print(f"Metrics saved to '{METRICS_FILE}'")


if __name__ == '__main__':
    main()
