#!/usr/bin/env python3
"""
Script to load a trained Gradient Boosting model and evaluate it on a population dataset.

Reads an Excel file containing predictor variables specified in `PREDICTORS` and
a binary target column `TARGET`. Drops rows with missing or infinite values row-by-row,
prints per-row predictions, and computes a comprehensive set of performance metrics.
Saves predictions and metrics to Excel files.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, log_loss, brier_score_loss, classification_report
)

# === User-configurable variables ===
# List of predictor column names in the Excel file
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
# Name of the target column
TARGET = 'Severe_all'
# Paths
EXCEL_FILE = 'iProve_gen_30.xlsx'      # Excel with individuals (rows) and columns
MODEL_FILE = 'gb_model_severe.joblib'           # Trained GB model file
# Threshold for class assignment from probability
THRESHOLD = 0.5

# Output files
PREDICTIONS_FILE = 'gb_predictions.xlsx'
METRICS_FILE = 'gb_metrics.xlsx'


def main():
    # Load data
    df = pd.read_excel(EXCEL_FILE)

    # Verify columns
    required_cols = PREDICTORS + [TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"All required columns present: {required_cols}")

    # Load model
    model = joblib.load(MODEL_FILE)
    print(f"Gradient Boosting model loaded from '{MODEL_FILE}'")

    # Prepare lists for metrics
    true_vals = []
    pred_probs = []
    pred_classes = []
    indices = []

    # Iterate row by row
    for idx, row in df.iterrows():
        # Skip missing or infinite
        if any(pd.isna(row[col]) for col in required_cols) or not np.isfinite([row[col] for col in required_cols]).all():
            print(f"Skipping row {idx}: missing or infinite data.")
            continue
        # Extract
        x = row[PREDICTORS].values.reshape(1, -1)
        y_true = row[TARGET]
        # Predict
        prob = model.predict_proba(x)[:,1][0]
        pred = int(prob >= THRESHOLD)
        # Print details
        predictor_dict = {col: row[col] for col in PREDICTORS}
        print(f"Row {idx}: Predictors={predictor_dict}, True={y_true}, PredProb={prob:.4f}, PredClass={pred}")
        # Store
        indices.append(idx)
        true_vals.append(y_true)
        pred_probs.append(prob)
        pred_classes.append(pred)

    # Convert to arrays
    if not true_vals:
        print("No valid rows to evaluate.")
        return
    y_true = np.array(true_vals)
    y_pred = np.array(pred_classes)
    y_prob = np.array(pred_probs)

    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp)>0 else np.nan,
        'Precision (PPV)': precision_score(y_true, y_pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn)>0 else np.nan,
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC ROC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan,
        'AUC PR': average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan,
        'Matthews CC': matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan,
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan,
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan,
        'Log Loss': log_loss(y_true, y_prob, labels=[0,1]) if len(np.unique(y_true))>1 else np.nan,
        'Brier Score': brier_score_loss(y_true, y_prob)
    }
    # Add confusion
    metrics.update({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})

    # Print overall metrics
    print("\nOverall Performance Metrics:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}" if isinstance(val, float) else f"{name}: {val}")

    # Classification report
    if len(np.unique(y_true))>1:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Class 0','Class 1'], zero_division=0))
    else:
        print(f"\nSkipping classification report: only one class present ({np.unique(y_true)[0]}).")

    # Save predictions
    out_df = df.loc[indices, PREDICTORS + [TARGET]].copy()
    out_df['Predicted_Prob'] = y_prob
    out_df['Predicted_Class'] = y_pred
    out_df.to_excel(PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to '{PREDICTIONS_FILE}'")

    # Save metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric','Value'])
    metrics_df.to_excel(METRICS_FILE, index=False)
    print(f"Metrics saved to '{METRICS_FILE}'")


if __name__ == '__main__':
    main()
