import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# === User-configurable variables ===
PREDICTORS = ['AirtestDico', 'edad', 'genero', 'IMC', 'ASA', 'spO2pre', 'DM', 'fumador', 'ePOC', 'fio2T3', 'duracionCirugia' ]  # <-- replace with your column names
TARGET = 'PPC_all'                            # target column name
EXCEL_FILE = 'iProve_gen_70.xlsx'
MAX_DEGREE = 10                                    # maximum polynomial degree
FINAL_FORMULA_FILE = 'final_polynomial_formula.xlsx'  # output Excel file for final formula
MODEL_OUTPUT_TEMPLATE = 'polynomial_model_deg.joblib'  # (optional) save best model
SAVE_BEST_MODEL = True                             # whether to save the best pipeline


def format_formula(coefs, intercept, feature_names):
    """
    Construct a human-readable formula string from coefficients and feature names.
    """
    terms = []
    # Intercept
    if abs(intercept) > 1e-12:
        terms.append(f"{intercept:.6g}")
    # Each coefficient term
    for coef, name in zip(coefs, feature_names):
        if abs(coef) < 1e-12:
            continue
        terms.append(f"{coef:.6g} * {name}")
    return " + ".join(terms)


def main():
    # Load and clean data
    df = pd.read_excel(EXCEL_FILE)
    required_cols = PREDICTORS + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("No data after dropping missing or infinite values.")
    X = df[PREDICTORS].values
    y = df[TARGET].values

    best = {'degree': None, 'r2': -np.inf, 'formula': None, 'pipeline': None}

    # Evaluate degrees
    for deg in range(1, MAX_DEGREE + 1):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('reg', LinearRegression())
        ])
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r2 = r2_score(y, y_pred)
        if r2 > best['r2']:
            feature_names = pipeline.named_steps['poly'].get_feature_names_out(PREDICTORS)
            coefs = pipeline.named_steps['reg'].coef_
            intercept = pipeline.named_steps['reg'].intercept_
            formula = format_formula(coefs, intercept, feature_names)
            best.update({'degree': deg, 'r2': r2, 'formula': formula, 'pipeline': pipeline})

    # Output final formula
    print(f"Best polynomial degree: {best['degree']}")
    print(f"R^2: {best['r2']:.6f}")
    print(f"Final formula: y = {best['formula']}")

    # Save final formula to Excel
    result_df = pd.DataFrame([{ 
        'degree': best['degree'],
        'r2': best['r2'],
        'formula': best['formula']
    }])
    result_df.to_excel(FINAL_FORMULA_FILE, index=False)
    print(f"Final formula saved to '{FINAL_FORMULA_FILE}'")

    # Optionally save best model pipeline
    if SAVE_BEST_MODEL and best['pipeline'] is not None:
        model_path = MODEL_OUTPUT_TEMPLATE.format(deg=best['degree'])
        joblib.dump(best['pipeline'], model_path)
        print(f"Best model pipeline saved to '{model_path}'")

if __name__ == '__main__':
    main()
