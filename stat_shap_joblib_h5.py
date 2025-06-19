#!/usr/bin/env python3
"""
explicabilidad_keras_excel.py

Explica una red neuronal Keras (.h5) con:
  • Permutation Importance
  • SHAP (DeepExplainer)
  • ICE de las variables top SHAP
y vuelca todo a un Excel con gráficas embebidas.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import shap, matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from tensorflow import keras
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from sklearn.metrics import make_scorer, accuracy_score, log_loss, mean_squared_error

# ---------- CONFIG ----------
DATA_FILE        = Path("iprove.xlsx")
SHEET_NAME       = 'Sheet1'
MODEL_FILE       = Path("trained_model.h5")
TARGET_COL       = "PPC_all"
PREDICTOR_COLS   = None
OUTPUT_XLSX      = Path("interpretaciones_red_n.xlsx")

N_SHAP_SAMPLE    = 5000
TOP_ICE_FEATURES = 3
N_ICE_SAMPLES    = 200
N_ICE_POINTS     = 20
RANDOM_STATE     = 0
# -----------------------------

def load_Xy():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    X = df.drop(columns=[TARGET_COL]) if PREDICTOR_COLS is None else df[PREDICTOR_COLS]
    y = df[TARGET_COL]
    return X, y

# ---------- predicción ----------
def model_predict(model, X_df):
    """Binaria: devuelve prob clase 1.  Multiclase: usa argmax."""
    proba = model.predict(X_df, verbose=0)
    if proba.shape[1] == 1:         # regresión o binaria con 1 salida
        return proba.ravel()
    if proba.shape[1] == 2:         # binaria con softmax 2-neurona
        return proba[:, 1]
    return proba.argmax(axis=1)     # multiclase → clase más probable

# ---------- métricas ----------
def perm_importance(model, X, y):
    """Permutation Importance compatible con modelos Keras."""

    if y is None:
        return pd.Series(dtype=float, name="permutation")

    # ---- Clasificación o regresión -----------------
    is_classif = pd.api.types.is_integer_dtype(y) and y.nunique() <= 20

    # scorer debe aceptar (estimator, X, y_true)
    def scorer(est, X_, y_true):
        y_pred = est.predict(X_, verbose=0)

        if is_classif:
            # --> convertir a matriz de probabilidades
            if y_pred.ndim == 1 or y_pred.shape[1] == 1:        # sigmoid
                y_pred = y_pred.reshape(-1, 1)
                y_pred = np.hstack((1 - y_pred, y_pred))
            score = -log_loss(y_true, y_pred)  # signo menos: mayor = mejor
        else:
            score = -mean_squared_error(y_true, y_pred)         # mayor = mejor

        return score

    r = permutation_importance(
        model, X, y,
        n_repeats=25,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring=scorer
    )

    return pd.Series(r.importances_mean, index=X.columns, name="permutation")

def shap_beeswarm(shap_obj, Xref):
    fig = plt.figure()
    shap.summary_plot(shap_obj, Xref, show=False)
    plt.title("SHAP summary (beeswarm)"); fig.tight_layout()
    return fig

def shap_importance(model, X):
    """
    Devuelve:
      • Serie con |SHAP| medio por variable
      • Objeto shap.Explanation (o array) para gráficas
      • DataFrame de referencia utilizado
    Maneja Deep, Gradient y Kernel según disponibilidad.
    """
    Xs = (X.sample(N_SHAP_SAMPLE, random_state=RANDOM_STATE)
          if N_SHAP_SAMPLE and len(X) > N_SHAP_SAMPLE else X)

    # -------- 1º intento: wrapper automático ------------
    try:
        explainer = shap.Explainer(model, Xs, feature_names=X.columns)
        sv = explainer(Xs)               # shap.Explanation
        vals = sv.values                 # (n, n_feat)
        if vals.shape[1] != X.shape[1]:
            raise ValueError("n_feat mismatch")  # forzamos fallback
        mean_abs = np.abs(vals).mean(axis=0)
        return (pd.Series(mean_abs, index=X.columns, name="mean_abs_shap"),
                sv, Xs)

    except Exception as e:
        print(f"[SHAP] Wrapper failed → {e}\n     Falling back…")

    # -------- 2º intento: GradientExplainer -------------
    try:
        grad_exp = shap.GradientExplainer(model, Xs.to_numpy())
        vals = grad_exp.shap_values(Xs.to_numpy())
        if isinstance(vals, list):       # clasificación
            vals = vals[0]
        if vals.shape[1] != X.shape[1]:
            raise ValueError("n_feat mismatch")
        mean_abs = np.abs(vals).mean(axis=0)
        return (pd.Series(mean_abs, index=X.columns, name="mean_abs_shap"),
                vals, Xs)

    except Exception as e:
        print(f"[SHAP] GradientExplainer failed → {e}\n     Using Kernel…")

    # -------- 3º intento (si todo lo demás falla): KernelExplainer ----------
    kernel_exp = shap.KernelExplainer(model_predict, Xs.iloc[:100, :])
    vals = kernel_exp.shap_values(Xs, nsamples=100)   # recórtalo si va lento
    if isinstance(vals, list):
        vals = vals[0]
    mean_abs = np.abs(vals).mean(axis=0)
    return (pd.Series(mean_abs, index=X.columns, name="mean_abs_shap"),
            vals, Xs)


# ---------- ICE ----------
def ice_df(model, X, feat, grid, ids):
    base = X.iloc[ids].copy()
    rows = []
    for v in grid:
        tmp = base.copy()
        tmp.loc[:, feat] = v
        rows.append(pd.DataFrame({
            "id": ids,
            "feat_val": v,
            "pred": model_predict(model, tmp)
        }))
    return pd.concat(rows, ignore_index=True)

def ice_plot(df, feat):
    fig, ax = plt.subplots()
    for _, g in df.groupby("id"):
        ax.plot(g["feat_val"], g["pred"], alpha=.2)
    ax.set_title(f"ICE – {feat}"); ax.set_xlabel(feat); ax.set_ylabel("Predicción")
    fig.tight_layout(); return fig

# ---------- gráficos aux ----------
def bar_plot(series, title, top=20):
    fig, ax = plt.subplots()
    s = series.sort_values(ascending=False).head(top)
    ax.barh(s.index[::-1], s[::-1])
    ax.set_title(title); ax.set_xlabel("Importancia")
    fig.tight_layout(); return fig

# ---------- main ----------
def main():
    X, y = load_Xy()
    model = keras.models.load_model(MODEL_FILE)

    imp_perm = perm_importance(model, X, y)
    imp_shap, sv_full, Xs = shap_importance(model, X)
    resumen = pd.concat([imp_perm, imp_shap], axis=1)
    resumen["rank_perm"] = resumen["permutation"].rank(ascending=False)
    resumen["rank_shap"] = resumen["mean_abs_shap"].rank(ascending=False)
    resumen.sort_values("rank_shap", inplace=True)

    figs = [
        bar_plot(imp_perm, "Permutation Importance"),
        bar_plot(imp_shap, "SHAP |mean| Importance"),
        shap_beeswarm(sv_full, Xs)
    ]

    rng = np.random.default_rng(RANDOM_STATE)
    ids = rng.choice(len(X), size=min(N_ICE_SAMPLES, len(X)), replace=False)
    top_feats = resumen.head(TOP_ICE_FEATURES).index.tolist()
    ice_dfs = {}
    for feat in top_feats:
        grid = np.quantile(X[feat], np.linspace(0,1,N_ICE_POINTS))
        df_ice = ice_df(model, X, feat, grid, ids)
        ice_dfs[feat] = df_ice
        figs.append(ice_plot(df_ice, feat))

    # -------- Excel --------
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as wr:
        resumen.to_excel(wr, sheet_name="resumen", index_label="feature")
        imp_perm.to_frame().to_excel(wr, sheet_name="permutation", index_label="feature")
        imp_shap.to_frame().to_excel(wr, sheet_name="shap", index_label="feature")
        for feat, df in ice_dfs.items():
            df.to_excel(wr, sheet_name=f"ice_{feat}", index=False)
        pd.DataFrame({"placeholder": []}).to_excel(wr, sheet_name="graficos", index=False)

    wb = load_workbook(OUTPUT_XLSX)
    ws = wb["graficos"]; row = 2
    for fig in figs:
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
        ws.add_image(XLImage(buf), f"B{row}")
        row += 20; plt.close(fig)
    wb.save(OUTPUT_XLSX)
    print(f"✔ Informe completo en {OUTPUT_XLSX.resolve()}")

if __name__ == "__main__":
    main()
