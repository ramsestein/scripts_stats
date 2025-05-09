import pandas as pd

def create_sheet_list_excel(input_excel, output_excel):
    """
    Crea un archivo Excel con una tabla que contiene la lista de pestañas del Excel proporcionado.

    :param input_excel: Ruta del archivo Excel de entrada.
    :param output_excel: Ruta del archivo Excel de salida.
    """
    try:
        # Leer las hojas del Excel de entrada
        excel_sheets = pd.ExcelFile(input_excel).sheet_names

        # Crear un DataFrame con los nombres de las hojas
        sheet_df = pd.DataFrame({'Sheet Names': excel_sheets})

        # Guardar el DataFrame en un nuevo archivo Excel
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            sheet_df.to_excel(writer, index=False, sheet_name='Sheet List')

        print(f"Archivo creado exitosamente: {output_excel}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejemplo de uso
input_excel_path = 'resultado.xlsx'  # Ruta al archivo de entrada
output_excel_path = 'archivo_salida.xlsx'  # Ruta al archivo de salida

create_sheet_list_excel(input_excel_path, output_excel_path)
