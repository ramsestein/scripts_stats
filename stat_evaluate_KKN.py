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
# Name of the target column to predict
TARGET = 'Severe_all'
# Path to Excel file
EXCEL_FILE = 'iprove_gen_30.xlsx'  
MODEL_FILE = 'knn_model_severe.joblib'                # trained k-NN model file
THRESHOLD = 0.5                                # probability threshold for classification

# Output files
PREDICTIONS_FILE = 'knn_predictions_severe.xlsx'
METRICS_FILE = 'knn_metrics_severe.xlsx'


def main():
    # Load data
    df = pd.read_excel(EXCEL_FILE)
    required = PREDICTORS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"All required columns present: {required}")

    # Load model
    model = joblib.load(MODEL_FILE)
    print(f"Loaded k-NN model from '{MODEL_FILE}'")

    # Prepare lists
    indices = []
    true_vals = []
    pred_probs = []
    pred_classes = []

    # Row-by-row evaluation
    for idx, row in df.iterrows():
        values = [row[col] for col in required]
        if any(pd.isna(values)) or not np.isfinite(values).all():
            print(f"Skipping row {idx}: missing or infinite data.")
            continue
        X_row = row[PREDICTORS].values.reshape(1, -1)
        y_true = row[TARGET]
        # Predict probability and class
        # k-NN predict_proba
        prob = model.predict_proba(X_row)[0, 1]
        pred = int(prob >= THRESHOLD)
        # Print per-row details
        predictor_dict = {col: row[col] for col in PREDICTORS}
        print(f"Row {idx}: Predictors={predictor_dict}, True={y_true}, PredProb={prob:.4f}, PredClass={pred}")
        indices.append(idx)
        true_vals.append(y_true)
        pred_probs.append(prob)
        pred_classes.append(pred)

    if not true_vals:
        print("No valid rows to evaluate.")
        return

    y_true = np.array(true_vals)
    y_pred = np.array(pred_classes)
    y_prob = np.array(pred_probs)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'Precision (PPV)': precision_score(y_true, y_pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC ROC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan,
        'AUC PR': average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan,
        'Matthews CC': matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan,
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan,
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan,
        'Log Loss': log_loss(y_true, y_prob, labels=[0,1]) if len(np.unique(y_true))>1 else np.nan,
        'Brier Score': brier_score_loss(y_true, y_prob)
    }
    metrics.update({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})

    # Print overall metrics
    print("\nOverall Performance Metrics:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}" if isinstance(val, float) else f"{name}: {val}")

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
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_excel(METRICS_FILE, index=False)
    print(f"Metrics saved to '{METRICS_FILE}'")


if __name__ == '__main__':
    main()
