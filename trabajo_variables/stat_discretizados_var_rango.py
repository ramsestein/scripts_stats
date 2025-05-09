import pandas as pd
import numpy as np

# Ruta al archivo Excel de entrada
input_path = "iProve_gen.xlsx"  # Cambia esta ruta si es necesario

# Leer el archivo Excel
df = pd.read_excel(input_path)

# Definir el array de discretización: [nombre_columna, mínimo, máximo]
variables_discretas = [
    ['edad', 40, 90, 10],
    ['peso', 40, 90, 10],
    ['hbPreoperatoria', 10, 18, 2],
    ['altura', 140, 190, 10],
    ['IMC', 15, 50, 5],
    ['peepFinal', 5, 20, 2],
    ['frFinal', 10, 25, 2],
    ['vtFinal', 200, 740, 20],
    ['Vt_ml_kg', 4, 12, 1],
    ['spo2Final',92, 99, 2],
    ['fio2T3', 40, 95, 10],
    ['pao2Final', 170, 450, 20],
    ['paco2Final', 30, 60, 5],
    ['presionMesetaFinal', 5, 25, 2],
    ['dPressure', 2, 30, 2],
    ['duracionCirugia', 20, 350, 20],
    ['cristaloides', 200, 3500, 50],
    ['Airtest', 92, 99, 2]
]

# Definir el array de variables a binarizar: [columna, umbral]
variables_binarias = [
    ['spO2pre', 97],
    ['Vt_ml_kg', 6.1],
    ['spo2Final', 97]
]

def dicotomized_value(variables_binarias):
    # Aplicar binarización: 1 si >= umbral, 0 si < umbral
    for var, threshold in variables_binarias:
        new_col = f"{var}_b"
        df[new_col] = df[var].apply(lambda x: 1 if pd.notnull(x) and x >= threshold else 0 if pd.notnull(x) else None)

# Función para discretizar un valor numérico
def discretize_value(value, min_val, max_val, rango):
    if pd.isnull(value):
        return np.nan
    if value < min_val:
        return min_val - rango
    elif value >= max_val:
        return max_val + rango
    else:
        for lower in range(min_val, max_val, rango):
            upper = lower + rango
            if lower <= value < upper:
                return (lower + upper) / 2
    return np.nan

if variables_discretas != []:
    for var, min_val, max_val, r in variables_discretas:
        new_col = f"{var}_d"
        df[new_col] = df[var].apply(lambda x: discretize_value(x, min_val, max_val,r))

if variables_binarias != []:
    dicotomized_value(variables_binarias)

# Guardar el nuevo DataFrame con las columnas discretizadas
output_path = "iProve_gen_discretizado.xlsx"
df.to_excel(output_path, index=False)

print(f"Archivo guardado como: {output_path}")
