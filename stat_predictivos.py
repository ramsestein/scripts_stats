import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from openpyxl import Workbook
import os

variables_imputar = ['edad', 'genero', 'IMC', 'ASA', 'fumador', 'tipoCirugia', 'SaFi_0', 'SaFi1', 'SaFi2', 
                     'SpO2_30', 'spO2pre(21)', 'SpO2_0', 'SpO2_1', 'SpO2_2', 'SpO2_7', 'SaFi_30', 'SaFi7', 
                     'difSaFi_1-2', 'airTest_SpO2']

# Cargar datos
data_path = "ESAIC_analisis.xlsx"
df_pre = pd.read_excel(data_path)
df = df_pre.dropna(thresh=5)

# Crear carpeta para resultados
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
output_excel = os.path.join(output_dir, "resultados_modelos.xlsx")

# Imputación por subgrupos
def imputar_por_grupo(df, group_col, impute_cols):
    imputado_df = df.copy()
    for col in impute_cols:
        for value in imputado_df[group_col].unique():
            grupo = imputado_df[imputado_df[group_col] == value]
            if grupo.empty:
                continue
            mediana = grupo[col].median()
            mas_frecuente = grupo[col].mode()[0] if not grupo[col].mode().empty else np.nan
            imputado_df.loc[grupo.index, col] = grupo[col].fillna(mas_frecuente)
            imputado_df.loc[grupo.index, col] = imputado_df.loc[grupo.index, col].fillna(mediana)
    return imputado_df

df = imputar_por_grupo(df, 'Composite infección', variables_imputar)

# Separar datos
X = df[variables_imputar]
y = df['Composite infección']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preparar workbook
wb = Workbook()

# Resultados
results = {}

# Regresión Logística
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
log_report = classification_report(y_test, y_pred_log, output_dict=True)
results['Regresión Logística'] = pd.DataFrame(log_report).T
print("Regresión Logística:")
print(results['Regresión Logística'])

# Random Forest con Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
cv_scores_rf = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
results['Random Forest'] = pd.DataFrame({"Fold": range(1, len(cv_scores_rf) + 1), "Accuracy": cv_scores_rf})
print("Random Forest:")
print(results['Random Forest'])

# XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)
results['XGBoost'] = pd.DataFrame(xgb_report).T
print("XGBoost:")
print(results['XGBoost'])

# Redes Neuronales
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
mlp_report = classification_report(y_test, y_pred_mlp, output_dict=True)
results['Redes Neuronales'] = pd.DataFrame(mlp_report).T
print("Redes Neuronales:")
print(results['Redes Neuronales'])

# Guardar en Excel
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    for key, value in results.items():
        if isinstance(value, str):
            pd.DataFrame([value]).to_excel(writer, sheet_name=key, index=False, header=False)
        else:
            value.to_excel(writer, sheet_name=key)

print(f"Resultados guardados en: {output_excel}")
