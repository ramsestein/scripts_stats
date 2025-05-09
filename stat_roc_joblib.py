import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# === User-configurable variables ===
# List any number of predictor column names from the Excel file
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
# Name of the target column to predict
TARGET = 'Severe_all'
# Excel file path
EXCEL_FILE = 'iProve_gen_30.xlsx'
# Gradient Boosting hyperparameters
# Output files
ROC_DATA_FILE = 'roc_curve_data.xlsx'
ROC_PLOT_FILE = 'roc_gb_severe.png'
# Output model file
MODEL_FILE = 'rf_model_severe.joblib'

def main():
    # Load and clean data
    df = pd.read_excel(EXCEL_FILE)
    required = PREDICTORS + [TARGET]
    # Verify columns exist
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Drop rows with NaN or infinite values
    df = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("No data left after dropping missing or infinite values.")

    # Extract features and labels
    X = df[PREDICTORS].values
    y_true = df[TARGET].values

    # Load model
    model = joblib.load(MODEL_FILE)
    print(f"Loaded model from '{MODEL_FILE}'")

    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Some models expose decision_function instead
        try:
            y_prob = model.decision_function(X)
        except AttributeError:
            raise RuntimeError("Model does not support probability prediction.")

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC ROC: {auc:.4f}")

    # Save ROC data to Excel
    roc_df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Threshold': thresholds
    })
    roc_df.to_excel(ROC_DATA_FILE, index=False)
    print(f"ROC curve data saved to '{ROC_DATA_FILE}'")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(ROC_PLOT_FILE)
    print(f"ROC curve plot saved to '{ROC_PLOT_FILE}'")
    plt.show()


if __name__ == '__main__':
    main()
