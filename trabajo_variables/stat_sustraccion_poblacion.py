import pandas as pd

# Nombres de los archivos
excel_1 = 'ESAIC_analisis.xlsx'  # Cambia el nombre si es necesario
excel_2 = 'ESAIC_SpO2_low2.xlsx'  # Cambia el nombre si es necesario
output_excel = 'ESAIC_SpO2_high2.xlsx'

# Cargar los datos
df1 = pd.read_excel(excel_1)
df2 = pd.read_excel(excel_2)

# Obtener los valores de Study+Patient del segundo archivo
study_patient_2 = set(df2['Study+Patient'])

# Filtrar las filas de df1 que no están en df2
filtradas = df1[~df1['Study+Patient'].isin(study_patient_2)]

# Eliminar filas con NaN en SpO2_1 o SpO2_2
filtradas = filtradas.dropna(subset=['SpO2_1', 'SpO2_2'])

# Guardar en un nuevo archivo Excel
filtradas.to_excel(output_excel, index=False)

print(f"Se han guardado {len(filtradas)} filas en el archivo {output_excel}.")
