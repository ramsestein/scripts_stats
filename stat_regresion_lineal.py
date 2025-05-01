import pandas as pd
import statsmodels.api as sm

# Cargar el archivo Excel
file_path = "ESAIC_AirTestSpO2.xlsx"
df = pd.read_excel(file_path)

# Filtrar valores no nulos para las variables de interés
df = df.dropna(subset=["SpO2_1", "SpO2_2", "airTest_SpO2"])

# Verificar si hay suficientes datos
if df.empty or len(df) < 2:
    raise ValueError("No hay suficientes datos para realizar la regresión.")

# Realizar regresión para SpO2_1
X_1 = df[["airTest_SpO2"]]
X_1 = sm.add_constant(X_1)  # Agregar constante
y_1 = df["SpO2_1"]

model_1 = sm.OLS(y_1, X_1).fit()

# Realizar regresión para SpO2_2
X_2 = df[["airTest_SpO2"]]
X_2 = sm.add_constant(X_2)  # Agregar constante
y_2 = df["SpO2_2"]

model_2 = sm.OLS(y_2, X_2).fit()

# Mostrar resúmenes
print("Regresión lineal para SpO2_1:")
print(model_1.summary())
print("\nRegresión lineal para SpO2_2:")
print(model_2.summary())

# Guardar resultados en un archivo Excel
output_path = "Resultados_Regresion_SpO2.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    pd.DataFrame(model_1.summary().tables[1]).to_excel(writer, sheet_name="Regresión_SpO2_1", index=False)
    pd.DataFrame(model_2.summary().tables[1]).to_excel(writer, sheet_name="Regresión_SpO2_2", index=False)

print(f"Resultados guardados en {output_path}")
