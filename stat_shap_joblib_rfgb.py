#!/usr/bin/env python3
"""
explicabilidad_excel_unico.py  – VERSIÓN SIN WARNING

Lee un solo Excel con target + predictores, calcula
importancias (árbol, permutación, SHAP) + ICE, genera
gráficos y lo guarda todo en interpretaciones.xlsx.

Requiere: pandas, joblib, shap, scikit-learn, matplotlib,
openpyxl, pillow
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

# ================== CONFIGURA AQUÍ ==================
DATA_FILE        = Path("iprove.xlsx")      # Excel con todos los datos
SHEET_NAME       = "Sheet1"                 # nombre o índice de la hoja
MODEL_FILE       = Path("rf_model.joblib")
TARGET_COL       = "PPC_all"                # columna objetivo
PREDICTOR_COLS   = ['edad_d', 'genero', 'altura_d', 'peso_d', 'IMC_d', 'cirugiaAbdominalProgramada', 'cirugiaAbdominalUrgencia', 'esCirugiaOncologica', 'ASA', 'aRISCAT', 'spO2pre_b', 'hbPreoperatoria_d', 'infeccionRespiratoriaUltimoMes', 'hipertensionArterial', 'DM', 'cardiopatiaIsquemica', 'fumador', 'consumoAlcohol', 'dislipemia', 'ePOC', 'insuficienciaRenal', 'peepFinal_d', 'frFinal_d', 'vtFinal_d', 'Vt_ml_kg_d', 'spo2Final_d', 'fio2T3_d', 'pao2Final_d', 'paco2Final_d', 'presionMesetaFinal_d', 'dPressure_d', 'cristaloides_d', 'duracionCirugia_d', 'usoFarmacosVasoactivos', 'rever1onRNM', 'epidural', 'maniobraRescateIntraoperatorio', 'maniobraRescatePostoperatorio', 'AirtestDico'] # <-- replace with your column names# None ⇒ todas menos target
OUTPUT_XLSX      = Path("interpretaciones_rf.xlsx")

# SHAP
N_SHAP_SAMPLE    = 5_000

# ICE
TOP_ICE_FEATURES = 3
N_ICE_SAMPLES    = 200
N_ICE_POINTS     = 20

RANDOM_STATE     = 0
# ====================================================


# ---------- CARGA DE DATOS ----------
def load_Xy():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    if PREDICTOR_COLS is None:
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df[PREDICTOR_COLS]
    y = df[TARGET_COL]
    return X, y


# ---------- PREDICCIÓN SEGURA ----------
def model_predict(model, X):
    """Devuelve predicción (o probabilidad) sin warnings."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p.max(axis=1)
    return model.predict(X)


# ---------- MÉTRICAS ----------
def tree_importance(model, feat_names):
    return pd.Series(model.feature_importances_, index=feat_names,
                     name="tree_importance")


def perm_importance(model, X, y):
    r = permutation_importance(model, X, y,
                               n_repeats=25,
                               random_state=RANDOM_STATE,
                               n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns,
                     name="permutation_importance")


def shap_importance(model, X):
    Xs = (X.sample(N_SHAP_SAMPLE, random_state=RANDOM_STATE)
          if N_SHAP_SAMPLE and len(X) > N_SHAP_SAMPLE else X)
    explainer = shap.Explainer(model, Xs, feature_names=X.columns)
    sv = explainer(Xs)
    vals = sv.values
    if vals.ndim == 3:                 # multiclase
        vals = np.abs(vals).mean(axis=(0, 2))
    else:
        vals = np.abs(vals).mean(axis=0)
    return pd.Series(vals, index=X.columns, name="mean_abs_shap"), sv


# ---------- ICE ----------
def ice_dataframe(model, X, feat, grid, ids):
    base = X.iloc[ids].copy()
    frames = []
    for v in grid:
        tmp = base.copy()
        tmp.loc[:, feat] = v
        frames.append(pd.DataFrame({
            "sample_id": ids,
            "feature_value": v,
            "pred": model_predict(model, tmp)
        }))
    return pd.concat(frames, ignore_index=True)


def ice_plot(df, feat):
    fig, ax = plt.subplots()
    for _, g in df.groupby("sample_id"):
        ax.plot(g["feature_value"], g["pred"], alpha=.2)
    ax.set_title(f"ICE – {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Predicción")
    fig.tight_layout()
    return fig


# ---------- GRÁFICOS AUX ----------
def bar_plot(series, title, top=20):
    fig, ax = plt.subplots()
    s = series.sort_values(ascending=False).head(top)
    ax.barh(s.index[::-1], s[::-1])
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def shap_beeswarm(sv, X):
    fig = plt.figure()
    shap.summary_plot(sv, X, show=False)
    plt.title("SHAP summary (beeswarm)")
    fig.tight_layout()
    return fig


# ---------- PRINCIPAL ----------
def main():
    # 1. Modelo y datos
    X, y = load_Xy()

    model = joblib.load(MODEL_FILE)

    # ---- PARCHE: añadimos los nombres si el modelo no los tiene
    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = np.array(X.columns, dtype=object)

    # 2. Importancias
    imp_tree = tree_importance(model, X.columns)
    imp_perm = perm_importance(model, X, y)
    imp_shap, sv = shap_importance(model, X)

    resumen = pd.concat([imp_tree, imp_perm, imp_shap], axis=1)
    resumen["rank_tree"] = resumen["tree_importance"].rank(ascending=False, method="min")
    resumen["rank_perm"] = resumen["permutation_importance"].rank(ascending=False, method="min")
    resumen["rank_shap"] = resumen["mean_abs_shap"].rank(ascending=False, method="min")
    resumen.sort_values("rank_shap", inplace=True)

    # 3. Gráficos globales
    figs = [
        bar_plot(imp_tree, "Model feature_importances_"),
        bar_plot(imp_perm, "Permutation Importance"),
        bar_plot(imp_shap, "SHAP |mean| Importance"),
        shap_beeswarm(sv, X)
    ]

    # 4. ICE
    rng = np.random.default_rng(RANDOM_STATE)
    ids = rng.choice(len(X), size=min(N_ICE_SAMPLES, len(X)), replace=False)
    top_feats = resumen.head(TOP_ICE_FEATURES).index.tolist()
    ice_dfs = {}
    for feat in top_feats:
        grid = np.quantile(X[feat], np.linspace(0, 1, N_ICE_POINTS))
        df_ice = ice_dataframe(model, X, feat, grid, ids)
        ice_dfs[feat] = df_ice
        figs.append(ice_plot(df_ice, feat))

    # 5. Guardar tablas
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as wr:
        resumen.to_excel(wr, sheet_name="resumen", index_label="feature")
        imp_tree.to_frame().to_excel(wr, sheet_name="tree_importance", index_label="feature")
        imp_perm.to_frame().to_excel(wr, sheet_name="permutation", index_label="feature")
        imp_shap.to_frame().to_excel(wr, sheet_name="shap", index_label="feature")
        for feat, df in ice_dfs.items():
            df.to_excel(wr, sheet_name=f"ice_{feat}", index=False)
        pd.DataFrame({"placeholder": []}).to_excel(wr, sheet_name="graficos", index=False)

    # 6. Incrustar imágenes
    wb = load_workbook(OUTPUT_XLSX)
    ws = wb["graficos"]
    row = 2
    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        ws.add_image(XLImage(buf), f"B{row}")
        plt.close(fig)
        row += 20
    wb.save(OUTPUT_XLSX)

    print(f"✔ Informe completo guardado en:\n  {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()

