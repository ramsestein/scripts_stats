import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# === User-configurable variables ===
# List any number of predictor column names from the Excel file
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
# Name of the target column to predict
TARGET = 'Severe_all'
# Excel file path
EXCEL_FILE = 'iProve_gen_70.xlsx'
# Neural network hyperparameters
NUM_LAYERS = 2       # Number of hidden layers
UNITS = 32           # Number of units per hidden layer
EPOCHS = 20          # Number of training epochs
BATCH_SIZE = 32      # Training batch size
VALIDATION = 0.02
# Output model file
MODEL_OUTPUT = 'trained_model_severe.h5'


def build_model(input_dim, num_layers, units):

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # binary output
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    # Load raw data
    df = pd.read_excel(EXCEL_FILE)

    # Check required columns
    required = PREDICTORS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")
    print(f"All required columns present: {required}")

    # Subset and clean
    df = df[required].copy()
    df = df.dropna()
    if df.empty:
        raise ValueError("No data left after dropping rows with missing values.")
    # Check for infinite values
    arr = df.values
    if not np.isfinite(arr).all():
        raise ValueError("Data contains infinite values; please clean or impute before training.")

    # Rename predictors for simplicity
    rename_dict = {old: f'var{idx+1}' for idx, old in enumerate(PREDICTORS)}
    df.rename(columns=rename_dict, inplace=True)
    predictor_vars = [rename_dict[c] for c in PREDICTORS]

    # Split features and target
    X = df[predictor_vars].values
    y = df[TARGET].values

    # Build and train
    model = build_model(input_dim=X.shape[1], num_layers=NUM_LAYERS, units=UNITS)
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION
    )

    # Save model
    model.save(MODEL_OUTPUT)
    print(f"Model trained ({NUM_LAYERS} layers, {UNITS} units, {EPOCHS} epochs, batch {BATCH_SIZE}) and saved to '{MODEL_OUTPUT}'.")


if __name__ == '__main__':
    main()
