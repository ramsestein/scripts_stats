import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.tools import add_constant
import os

# Cargar datos
file_path = "AirtestDico_1_rest.xlsx"  # Ruta del archivo
data = pd.read_excel(file_path)

# Crear carpeta para guardar resultados
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Limpieza de datos
# 1. Eliminar filas con demasiados valores faltantes (menos de 5 variables válidas).
min_valid_columns = 5
data = data.dropna(thresh=min_valid_columns)

# 2. Imputación por subgrupos según 'Composite infección'
for col in data.columns:
    if data[col].isnull().sum() > 0 and col != "Composite infección":
        # Calcular mediana y moda solo para valores numéricos
        if pd.api.types.is_numeric_dtype(data[col]):
            median_1 = data.loc[data["Composite infección"] == 1, col].median()
            mode_1 = data.loc[data["Composite infección"] == 1, col].mode().iloc[0] if not data.loc[data["Composite infección"] == 1, col].mode().empty else np.nan
            value_1 = median_1 if pd.notna(median_1) else mode_1

            median_0 = data.loc[data["Composite infección"] == 0, col].median()
            mode_0 = data.loc[data["Composite infección"] == 0, col].mode().iloc[0] if not data.loc[data["Composite infección"] == 0, col].mode().empty else np.nan
            value_0 = median_0 if pd.notna(median_0) else mode_0

            data.loc[(data["Composite infección"] == 1) & (data[col].isnull()), col] = value_1
            data.loc[(data["Composite infección"] == 0) & (data[col].isnull()), col] = value_0
        else:
            # Para columnas no numéricas, imputar con el valor más frecuente del subgrupo
            mode_1 = data.loc[data["Composite infección"] == 1, col].mode().iloc[0] if not data.loc[data["Composite infección"] == 1, col].mode().empty else np.nan
            mode_0 = data.loc[data["Composite infección"] == 0, col].mode().iloc[0] if not data.loc[data["Composite infección"] == 0, col].mode().empty else np.nan
            data.loc[(data["Composite infección"] == 1) & (data[col].isnull()), col] = mode_1
            data.loc[(data["Composite infección"] == 0) & (data[col].isnull()), col] = mode_0

# Confounders definidos por el usuario
confounders = ["edad", "genero", "IMC", "ASA", "fumador", "tipoCirugia"]

# Selección de variables para el modelo (además de confounders)
predictors = [
    "edad", "genero", "IMC", "ASA", "fumador", "tipoCirugia",
    "SaFi_0", "SaFi1", "SaFi2", "SpO2_30", "spO2pre(21)",
    "SpO2_0", "SpO2_1", "SpO2_2", "SpO2_7", "SaFi_30",
    "SaFi7", "difSaFi_1-2", "airTest_SpO2"
]

# Eliminar predictores con alta correlación pero manteniendo los confounders
correlation_matrix = data[predictors].corr().abs()
high_corr = correlation_matrix > 0.9
to_drop = [column for column in high_corr.columns if any(high_corr[column]) and column not in confounders]
predictors = [col for col in predictors if col not in to_drop]

# Asegurar que los confounders estén presentes en los predictores
for conf in confounders:
    if conf not in predictors:
        predictors.append(conf)

# Preparar datos para el modelo
X = data[predictors]
X_const = add_constant(X)  # Agregar término constante
y = data["Composite infección"]
groups = data["tipoCirugia"]

# Crear y ajustar el modelo GEE
family = Binomial()
cov_struct = Exchangeable()

try:
    model_gee = GEE(y, X_const, groups=groups, family=family, cov_struct=cov_struct)
    result = model_gee.fit()

    # Mostrar resultados en terminal
    print("Modelo GEE ajustado:")
    print(result.summary())

    # Guardar resultados en un archivo Excel
    results_summary = pd.DataFrame({
        "Variable": result.params.index,
        "Coeficiente": result.params.values,
        "Error estándar": result.bse.values,
        "Z-valor": result.tvalues.values,
        "P-valor": result.pvalues.values,
        "Intervalo bajo": result.conf_int().iloc[:, 0].values,
        "Intervalo alto": result.conf_int().iloc[:, 1].values
    })

    results_summary.to_excel(os.path.join(output_dir, "gee_results_rest.xlsx"), index=False)
    print(f"Resultados guardados en {os.path.join(output_dir, 'gee_results.xlsx')}")

except Exception as e:
    print(f"Error al ajustar el modelo GEE: {e}")
