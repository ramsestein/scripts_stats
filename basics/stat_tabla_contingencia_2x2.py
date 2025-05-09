import pandas as pd

# 1. Carga de datos
ruta_fichero = 'iProve_gen_30.xlsx'     # Ajusta la ruta si hace falta
df = pd.read_excel(ruta_fichero)
test = 'AirtestDico'
resultado = 'PPC_all'

# 2. Tabla de contingencia
# Asegúrate de usar el nombre exacto de la columna: en tu caso 'AirtestDico'
contingencia = pd.crosstab(
    df[test],   # filas: categorías de AirtestDico
    df[resultado],       # columnas: 0 y 1 de PPC_all
    rownames=[test],
    colnames=[resultado],
    margins=True,        # añade totales
    margins_name='Total'
)

# 3. Mostrar por pantalla
print("Tabla de contingencia AirtestDico × PPC_all:\n")
print(contingencia)

# 4. (Opcional) Guardar la tabla en un nuevo Excel
contingencia.to_excel('tabla_contingencia_AirtestDico_PPC_all.xlsx')
print("\nTabla guardada en 'tabla_contingencia_AirtestDico_PPC_all.xlsx'")
