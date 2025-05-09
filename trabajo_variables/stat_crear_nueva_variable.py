import pandas as pd

def add_lowest_saturation(input_excel, output_excel, sheet_name=None):
    """
    Lee un archivo Excel, compara las columnas SpO2_1 y SpO2_2, y añade una nueva columna con la saturación más baja.
    Guarda el resultado en un nuevo archivo Excel.

    :param input_excel: Ruta al archivo Excel de entrada.
    :param output_excel: Ruta al archivo Excel de salida.
    :param sheet_name: Nombre de la hoja a procesar. Si es None, se usará la primera hoja.
    """
    try:
        # Leer el archivo Excel
        if sheet_name:
            df = pd.read_excel(input_excel, sheet_name=sheet_name)
        else:
            df = pd.read_excel(input_excel)

        # Verificar si las columnas necesarias existen
        if 'SpO2_1' not in df.columns or 'SpO2_2' not in df.columns:
            raise ValueError("El archivo Excel no contiene las columnas 'SpO2_1' y/o 'SpO2_2'.")

        # Crear la nueva columna con la saturación más baja
        df['Saturation_Lowest'] = df[['SpO2_1', 'SpO2_2']].min(axis=1)

        # Guardar el resultado en un nuevo archivo Excel
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        print(f"Archivo creado exitosamente con la nueva columna: {output_excel}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejemplo de uso
input_excel_path = 'ESAIC_analisis.xlsx'  # Ruta al archivo Excel de entrada
output_excel_path = 'archivo_salida.xlsx'  # Ruta al archivo Excel de salida

add_lowest_saturation(input_excel_path, output_excel_path)
