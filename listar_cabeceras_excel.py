import pandas as pd

# Cambia 'archivo.xlsx' por el nombre de tu archivo Excel
archivo_excel = 'iProve_gen.xlsx'

# Lee únicamente la primera fila (cabeceras)
df = pd.read_excel(archivo_excel, nrows=0)

# Extrae las cabeceras y conviértelas en una lista
cabeceras = list(df.columns)

# Imprime el array de cabeceras
print("Cabeceras de la tabla:", cabeceras)
