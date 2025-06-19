#!/usr/bin/env python3
"""
explicabilidad_xgboost_excel.py
--------------------------------
Explica un modelo XGBoost (.joblib o .json/.model convertidos a Booster)
y guarda tablas + gráficos en interpretaciones.xlsx
"""

from pathlib import Path
import pandas as pd
import numpy as np
import shap, matplotlib.pyplot as plt, xgboost as xgb, joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, mean_squared_error
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

# ======== CONFIGURA AQUÍ ========
DATA_FILE        = Path("iprove.xlsx")
SHEET_NAME       = 'Sheet1'
MODEL_FILE       = Path("xgb_model.joblib")
TARGET_COL       = "PPC_all"
PREDICTOR_COLS   = None
OUTPUT_XLSX      = Path("interpretaciones_xgb.xlsx")

N_SHAP_SAMPLE    = 5000
TOP_ICE_FEATURES = 3
N_ICE_SAMPLES    = 200
N_ICE_POINTS     = 20
RANDOM_STATE     = 0
# =================================


# ====================== UTILIDADES BÁSICAS ====================
def load_Xy():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    X = df.drop(columns=[TARGET_COL]) if PREDICTOR_COLS is None else df[PREDICTOR_COLS]
    y = df[TARGET_COL]
    return X, y


def load_xgb_model(path):
    """Devuelve un estimador scikit-learn (envuelve Booster si es necesario)."""
    mdl = joblib.load(path)

    if isinstance(mdl, xgb.Booster):            # Booster “puro”
        wrapper = xgb.XGBClassifier()           # dummy wrapper
        wrapper._Booster = mdl
        wrapper.n_classes_ = 2                  # ajusta si multiclase
        mdl = wrapper

    return mdl


def model_predict(model, X_df):
    """Predicción segura: siempre recibe DataFrame, internamente usa NumPy."""
    X_np = X_df.to_numpy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_np)
        if proba.shape[1] == 2:                 # binaria
            return proba[:, 1]
        return proba                            # multiclase
    return model.predict(X_np)                  # regresión
# ==============================================================


# ====================== MÉTRICAS GLOBALES =====================
def perm_importance(model, X, y):
    """Permutation Importance válido para clasificación o regresión."""
    if y is None:
        return pd.Series(dtype=float, name="permutation")

    is_classif = pd.api.types.is_integer_dtype(y) and y.nunique() <= 20
    columns = X.columns                        # cierre para scorer

    def scorer(est, X_np, y_true):
        X_df = pd.DataFrame(X_np, columns=columns)
        y_pred = model_predict(est, X_df)

        if is_classif:
            if y_pred.ndim == 1:               # convertir sigmoid → 2-col
                y_pred = np.vstack((1 - y_pred, y_pred)).T
            return -log_loss(y_true, y_pred)   # mayor = mejor
        return -mean_squared_error(y_true, y_pred)

    r = permutation_importance(
        model, X.to_numpy(), y,
        n_repeats=25, n_jobs=-1, random_state=RANDOM_STATE,
        scoring=scorer
    )
    return pd.Series(r.importances_mean, index=columns, name="permutation")


def shap_importance(model, X):
    """|SHAP| medio por variable + valores completos."""
    Xs = X.sample(N_SHAP_SAMPLE, random_state=RANDOM_STATE) \
           if N_SHAP_SAMPLE and len(X) > N_SHAP_SAMPLE else X
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)            # array (n, n_feat) o list
    if isinstance(sv, list):                  # clasificación multiclase
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns, name="mean_abs_shap"), sv, Xs
# ==============================================================


# ====================== ICE Y GRÁFICAS ========================
def ice_dataframe(model, X, feat, grid, idx):
    base = X.iloc[idx].copy()
    out = []
    for v in grid:
        tmp = base.copy()
        tmp.loc[:, feat] = v
        out.append(pd.DataFrame({
            "sample_id": idx,
            "feat_val": v,
            "pred": model_predict(model, tmp)
        }))
    return pd.concat(out, ignore_index=True)


def ice_plot(df, feat):
    fig, ax = plt.subplots()
    for _, g in df.groupby("sample_id"):
        ax.plot(g["feat_val"], g["pred"], alpha=.2)
    ax.set_xlabel(feat); ax.set_ylabel("Predicción")
    ax.set_title(f"ICE – {feat}"); fig.tight_layout()
    return fig


def bar_plot(series, title, top=20):
    fig, ax = plt.subplots()
    s = series.sort_values(ascending=False).head(top)
    ax.barh(s.index[::-1], s[::-1])
    ax.set_xlabel("Importancia"); ax.set_title(title)
    fig.tight_layout(); return fig


def shap_beeswarm(shap_vals, Xref):
    fig = plt.figure()
    shap.summary_plot(shap_vals, Xref, show=False)
    plt.title("SHAP summary (beeswarm)")
    fig.tight_layout(); return fig
# ==============================================================


# ============================ MAIN ============================
def main():
    # 1) Datos y modelo
    X, y = load_Xy()
    model = load_xgb_model(MODEL_FILE)

    # 2) Métricas globales
    imp_perm = perm_importance(model, X, y)
    imp_shap, shap_values, X_ref = shap_importance(model, X)

    resumen = pd.concat([imp_perm, imp_shap], axis=1)
    resumen["rank_perm"] = resumen["permutation"].rank(ascending=False)
    resumen["rank_shap"] = resumen["mean_abs_shap"].rank(ascending=False)
    resumen.sort_values("rank_shap", inplace=True)

    # 3) Gráficas globales
    figs = [
        bar_plot(imp_perm, "Permutation Importance"),
        bar_plot(imp_shap, "SHAP |mean| Importance"),
        shap_beeswarm(shap_values, X_ref)
    ]

    # 4) ICE
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(N_ICE_SAMPLES, len(X)), replace=False)
    top_feats = resumen.head(TOP_ICE_FEATURES).index.tolist()
    ice_dfs = {}
    for feat in top_feats:
        grid = np.quantile(X[feat], np.linspace(0, 1, N_ICE_POINTS))
        df_ice = ice_dataframe(model, X, feat, grid, idx)
        ice_dfs[feat] = df_ice
        figs.append(ice_plot(df_ice, feat))

    # 5) Guardar Excel con tablas
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as wr:
        resumen.to_excel(wr, sheet_name="resumen", index_label="feature")
        imp_perm.to_frame().to_excel(wr, sheet_name="permutation", index_label="feature")
        imp_shap.to_frame().to_excel(wr, sheet_name="shap", index_label="feature")
        for feat, df in ice_dfs.items():
            df.to_excel(wr, sheet_name=f"ice_{feat}", index=False)
        pd.DataFrame({"placeholder": []}).to_excel(wr, sheet_name="graficos", index=False)

    # 6) Incrustar imágenes en hoja 'graficos'
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
