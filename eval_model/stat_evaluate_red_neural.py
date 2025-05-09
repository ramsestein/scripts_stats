import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score,
    roc_auc_score, f1_score, classification_report
)

# === User-configurable variables ===
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
TARGET = 'Severe_all'                            # target column name
EXCEL_FILE = 'iProve_gen_30.xlsx'
MODEL_FILE = 'trained_model_severe.h5'               # path to trained .h5 model
THRESHOLD = 0.5                               # classification threshold for positive class

# Output files
PREDICTIONS_FILE = 'model_nn_predictions_severe.xlsx'
METRICS_FILE = 'model_nn_metrics_severe.xlsx'


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
    model = load_model(MODEL_FILE)

    # Lists for aggregation
    true_vals = []
    pred_probs = []
    pred_classes = []
    row_indices = []

    # Iterate row by row
    for idx, row in df.iterrows():
        # Check for missing in predictors or target
        if any(pd.isna(row[col]) for col in required_cols):
            print(f"Skipping row {idx}: missing data in predictors or target.")
            continue
        # Extract predictor values
        predictor_values = [row[col] for col in PREDICTORS]
        true_val = row[TARGET]

        # Prepare input and predict
        X_row = np.array(predictor_values).reshape(1, -1)
        prob = model.predict(X_row, verbose=0).ravel()[0]
        pred_class = int(prob >= THRESHOLD)

        # Print per-row detail
        details = {col: row[col] for col in PREDICTORS}
        print(f"Row {idx}: Predictors={details}, True={true_val}, Predicted_prob={prob:.4f}, Predicted_class={pred_class}")

        # Aggregate
        row_indices.append(idx)
        true_vals.append(true_val)
        pred_probs.append(prob)
        pred_classes.append(pred_class)

    # Convert to arrays
    if not true_vals:
        print("No valid rows to evaluate.")
        return
    y_true = np.array(true_vals)
    y_pred = np.array(pred_classes)
    y_prob = np.array(pred_probs)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'PPV (Precision)': precision_score(y_true, y_pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['AUC'] = np.nan
    metrics.update({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})

    # Print overall metrics
    print("\nOverall Performance Metrics:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    # Classification report if applicable
    unique_classes = np.unique(y_true)
    if unique_classes.size == 2:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Class 0','Class 1'], zero_division=0))
    else:
        print(f"\nSkipping classification report: only one class present ({unique_classes[0]}).")

    # Save predictions
    out_df = df.loc[row_indices, PREDICTORS + [TARGET]].copy()
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
