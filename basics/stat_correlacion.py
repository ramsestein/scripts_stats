import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = "PPC_all.xlsx"  # Cambia esto por la ruta a tu archivo
df = pd.read_excel(file_path)

# Eliminar columnas no numéricas (opcional, basado en tu dataset)
df_numeric = df.select_dtypes(include=['number'])

# Calcular la matriz de correlación
correlation_matrix = df_numeric.corr()

# Guardar la matriz de correlación en un archivo Excel
output_file = "correlation_matrix_PPC1.xlsx"
correlation_matrix.to_excel(output_file)
print(f"Matriz de correlación guardada en: {output_file}")

# Crear un mapa de calor para visualizar la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Mapa de calor de correlación")
plt.tight_layout()

# Guardar el gráfico como una imagen
heatmap_file = "correlation_heatmap_PPC1.png"
plt.savefig(heatmap_file)
print(f"Mapa de calor guardado en: {heatmap_file}")
plt.show()
