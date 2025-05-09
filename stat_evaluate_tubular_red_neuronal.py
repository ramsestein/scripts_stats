import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, log_loss, brier_score_loss, classification_report
)

# === User-configurable variables ===
PREDICTORS = ['gb', 'knn', 'nn', 'naive', 'rf', 'xgb']  # <-- replace with your column names
TARGET = 'PPC_all'                             # target column name
EXCEL_FILE = 'gb_metrics.xlsx'                  # dataset path
MODEL_FILE = 'deep_tabular_nn_train.h5'               # trained deep NN model file
THRESHOLD = 0.5                                 # probability threshold for classification

# Output files
PREDICTIONS_FILE = 'deep_nn_predictions_trained.xlsx'
METRICS_FILE = 'deep_nn_metrics_valida.xlsx'


def main():
    # Load data
    df = pd.read_excel(EXCEL_FILE)
    required = PREDICTORS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"All required columns present: {required}")

    # Load model
    model = load_model(MODEL_FILE)
    print(f"Loaded deep tabular NN model from '{MODEL_FILE}'")

    # Prepare lists
    indices = []
    true_vals = []
    pred_probs = []
    pred_classes = []

    # Iterate row-by-row
    for idx, row in df.iterrows():
        values = [row[col] for col in required]
        if any(pd.isna(values)) or not np.isfinite(values).all():
            print(f"Skipping row {idx}: missing or infinite data.")
            continue
        X_row = np.array(row[PREDICTORS].values, dtype=float).reshape(1, -1)
        y_true = row[TARGET]
        # Predict probability and class
        prob = model.predict(X_row, verbose=0).ravel()[0]
        pred = int(prob >= THRESHOLD)
        # Print per-row details
        predictor_dict = {col: row[col] for col in PREDICTORS}
        print(f"Row {idx}: Predictors={predictor_dict}, True={y_true}, PredProb={prob:.4f}, PredClass={pred}")
        indices.append(idx)
        true_vals.append(y_true)
        pred_probs.append(prob)
        pred_classes.append(pred)

    # Check any valid rows
    if not true_vals:
        print("No valid rows to evaluate.")
        return

    # Convert to arrays
    y_true = np.array(true_vals)
    y_pred = np.array(pred_classes)
    y_prob = np.array(pred_probs)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'Precision (PPV)': precision_score(y_true, y_pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'Matthews CC': matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        'Log Loss': log_loss(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'Brier Score': brier_score_loss(y_true, y_prob)
    }
    metrics.update({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})

    # Print overall metrics
    print("\nOverall Performance Metrics:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}" if isinstance(val, float) else f"{name}: {val}")

    # Classification report
    if len(np.unique(y_true)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Class 0','Class 1'], zero_division=0))
    else:
        print(f"\nSkipping classification report: only one class present ({np.unique(y_true)[0]})")

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
