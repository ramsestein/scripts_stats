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
MODEL_FILE       = Path("naive_bayes_model.joblib")
TARGET_COL       = "PPC_all"
PREDICTOR_COLS   = None
OUTPUT_XLSX      = Path("interpretaciones_nb.xlsx")

N_SHAP_SAMPLE    = 5000
TOP_ICE_FEATURES = 3
N_ICE_SAMPLES    = 200
N_ICE_POINTS     = 20
RANDOM_STATE     = 0
# =================================

# ===================== UTILIDADES ===========================
def load_Xy():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    X = df.drop(columns=[TARGET_COL]) if PREDICTOR_COLS is None else df[PREDICTOR_COLS]
    y = df[TARGET_COL]
    return X, y


def load_model(path):
    return joblib.load(path)


def model_predict(model, X_df):
    """Devuelve prob de clase 1 (binaria) o matriz de probas (multiclase)"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    return model.predict(X_df)
# ============================================================


# ================ PERMUTATION IMPORTANCE ====================
from sklearn.metrics import make_scorer

def perm_importance(model, X, y):
    is_class = pd.api.types.is_integer_dtype(y)

    def scorer(est, X_, y_true):
        y_pred = model_predict(est, pd.DataFrame(X_, columns=X.columns))
        if y_pred.ndim == 1:            # binaria → ya prob de clase 1
            ll = -log_loss(y_true, np.vstack((1 - y_pred, y_pred)).T)
            return ll
        if y_pred.ndim == 2:            # multiclase
            return -log_loss(y_true, y_pred)
        # regresión (no típico en NB) — accuracy no aplica
        return -((y_true - y_pred) ** 2).mean()

    r = permutation_importance(
        model, X, y,
        n_repeats=25,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring=scorer
    )
    return pd.Series(r.importances_mean, index=X.columns, name="permutation")
# ============================================================


# ====================== SHAP ================================
def shap_importance(model, X):
    Xs = X.sample(N_SHAP_SAMPLE, random_state=RANDOM_STATE) \
           if N_SHAP_SAMPLE and len(X) > N_SHAP_SAMPLE else X
    try:
        explainer = shap.LinearExplainer(model, Xs, feature_perturbation="interventional")
        sv = explainer.shap_values(Xs)
    except Exception:
        # ---------- Fallback robusto ----------
        f = lambda a: model_predict(
                model,
                pd.DataFrame(a, columns=X.columns)
            )
        explainer = shap.KernelExplainer(f, Xs.iloc[:100, :].to_numpy())
        sv = explainer.shap_values(Xs.to_numpy(), nsamples=100)
    # ----------------------------------------
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns, name="mean_abs_shap"), sv, Xs


# ================= ICE Y GRÁFICAS ===========================
def ice_dataframe(model, X, feat, grid, idx):
    base = X.iloc[idx].copy()
    frames = []
    for v in grid:
        tmp = base.copy()
        tmp.loc[:, feat] = v
        frames.append(pd.DataFrame({
            "id": idx,
            "feat_val": v,
            "pred": model_predict(model, tmp)
        }))
    return pd.concat(frames, ignore_index=True)


def ice_plot(df, feat):
    fig, ax = plt.subplots()
    for _, g in df.groupby("id"):
        ax.plot(g["feat_val"], g["pred"], alpha=.2)
    ax.set_xlabel(feat); ax.set_ylabel("Predicción o proba")
    ax.set_title(f"ICE – {feat}"); fig.tight_layout()
    return fig


def bar_plot(series, title, top=20):
    fig, ax = plt.subplots()
    s = series.sort_values(ascending=False).head(top)
    ax.barh(s.index[::-1], s[::-1])
    ax.set_title(title); ax.set_xlabel("Importancia")
    fig.tight_layout(); return fig


def shap_beeswarm(sv, Xref):
    fig = plt.figure()
    shap.summary_plot(sv, Xref, show=False)
    plt.title("SHAP summary (beeswarm)")
    fig.tight_layout(); return fig
# ============================================================


# ============================ MAIN ==========================
def main():
    X, y = load_Xy()
    model = load_model(MODEL_FILE)

    # 1) Importancias
    imp_perm = perm_importance(model, X, y)
    imp_shap, shap_vals, X_ref = shap_importance(model, X)

    resumen = pd.concat([imp_perm, imp_shap], axis=1)
    resumen["rank_perm"] = resumen["permutation"].rank(ascending=False)
    resumen["rank_shap"] = resumen["mean_abs_shap"].rank(ascending=False)
    resumen.sort_values("rank_shap", inplace=True)

    # 2) Gráficas globales
    figs = [
        bar_plot(imp_perm, "Permutation Importance"),
        bar_plot(imp_shap, "SHAP |mean| Importance"),
        shap_beeswarm(shap_vals, X_ref)
    ]

    # 3) ICE
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(N_ICE_SAMPLES, len(X)), replace=False)
    top_feats = resumen.head(TOP_ICE_FEATURES).index.tolist()
    ice_dfs = {}
    for feat in top_feats:
        grid = np.quantile(X[feat], np.linspace(0, 1, N_ICE_POINTS))
        df_ice = ice_dataframe(model, X, feat, grid, idx)
        ice_dfs[feat] = df_ice
        figs.append(ice_plot(df_ice, feat))

    # 4) Guardar Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as wr:
        resumen.to_excel(wr, sheet_name="resumen", index_label="feature")
        imp_perm.to_frame().to_excel(wr, sheet_name="permutation", index_label="feature")
        imp_shap.to_frame().to_excel(wr, sheet_name="shap", index_label="feature")
        for feat, df in ice_dfs.items():
            df.to_excel(wr, sheet_name=f"ice_{feat}", index=False)
        pd.DataFrame({"placeholder": []}).to_excel(wr, sheet_name="graficos", index=False)

    # 5) Incrustar imágenes
    wb = load_workbook(OUTPUT_XLSX)
    ws = wb["graficos"]; row = 2
    for fig in figs:
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
        ws.add_image(XLImage(buf), f"B{row}")
        plt.close(fig); row += 20
    wb.save(OUTPUT_XLSX)

    print(f"✔ Informe completo guardado en {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
