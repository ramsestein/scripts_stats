import pandas as pd

def filter_excel(input_file, output_file):
    """
    Crea un nuevo Excel con solo las filas que tienen contenido en SpO2_1 y SpO2_2.

    :param input_file: Ruta del archivo Excel de entrada.
    :param output_file: Ruta del archivo Excel de salida.
    """
    # Leer el archivo Excel
    df = pd.read_excel(input_file)
    
    # Filtrar filas con valores no vacíos en SpO2_1 y SpO2_2
    filtered_df = df.dropna(subset=['SpO2_1', 'SpO2_2'])
    
    # Guardar el resultado en un nuevo archivo Excel
    filtered_df.to_excel(output_file, index=False)
    print(f"Archivo filtrado guardado en: {output_file}")

# Ejemplo de uso
filter_excel(
    input_file="ESAIC_analisis.xlsx",
    output_file="ESAIC_SpO2_1_2.xlsx"
)
