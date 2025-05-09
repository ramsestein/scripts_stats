import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# === User-configurable variables ===
# List of predictor column names in the Excel file
PREDICTORS = ['gb', 'knn', 'nn', 'naive', 'rf', 'xgb']  # <-- replace with your column names
# Name of the target column to predict
TARGET = 'PPC_all'
# Path to Excel file
EXCEL_FILE = 'gb_metrics.xlsx'

# Neural network hyperparameters
HIDDEN_UNITS = [64, 32, 16]  # list specifying number of units in each hidden layer
DROPOUT_RATE = 0.2           # dropout fraction between layers
LEARNING_RATE = 0.001        # optimizer learning rate
EPOCHS = 20                 # training epochs
BATCH_SIZE = 32              # size of mini-batches
VALIDATION_SPLIT = 0.02       # fraction of data for validation

# Output model file
MODEL_OUTPUT = 'deep_tabular_nn_train.h5'


def build_model(input_dim, hidden_units, dropout_rate, learning_rate):
    """
    Builds a deep feedforward neural network for binary classification.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


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
    arr = df.values
    if not np.isfinite(arr).all():
        raise ValueError("Data contains infinite values; clean or impute before training.")

    # Extract features and labels
    X = df[PREDICTORS].values
    y = df[TARGET].values

    # Build and train model
    model = build_model(
        input_dim=X.shape[1],
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    # Save the trained model
    model.save(MODEL_OUTPUT)
    print(f"Deep tabular NN trained and saved to '{MODEL_OUTPUT}'")


if __name__ == '__main__':
    main()
