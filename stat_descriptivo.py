# Importar librerías necesarias
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook

# Función para cargar datos y realizar limpieza inicial
def cargar_datos(file_path, columns_of_interest, umbral=5):
    """
    Carga el archivo de datos y elimina filas con valores nulos en exceso.
    
    Args:
    - file_path (str): Ruta al archivo de datos.
    - columns_of_interest (list): Columnas clave para análisis.
    - umbral (int): Mínimo de valores no nulos requeridos por fila.
    
    Returns:
    - pd.DataFrame: DataFrame limpio.
    """
    try:
        df = pd.ExcelFile(file_path).parse('Sheet1')
        df_clean = df.dropna(subset=columns_of_interest, thresh=umbral)
        return df_clean
    except Exception as e:
        print(f"Error al cargar o procesar el archivo: {e}")
        return None

# Función para mostrar estadísticas descriptivas
def estadisticas_descriptivas(df, columns):
    """
    Calcula estadísticas descriptivas de las columnas especificadas.
    
    Args:
    - df (pd.DataFrame): DataFrame limpio.
    - columns (list): Columnas clave para análisis.
    
    Returns:
    - pd.DataFrame: DataFrame con estadísticas descriptivas.
    """
    desc_stats = df[columns].describe()
    print("\nEstadísticas descriptivas de las variables clave:")
    print(desc_stats)
    return desc_stats

# Función para visualizar y guardar distribuciones
def visualizar_distribuciones(df, columns, output_dir):
    """
    Genera boxplots de cada columna según la presencia de infección (Composite infección)
    y guarda las gráficas en una carpeta.
    
    Args:
    - df (pd.DataFrame): DataFrame limpio.
    - columns (list): Columnas clave para visualización.
    - output_dir (str): Ruta de la carpeta donde guardar las gráficas.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for column in columns:
        if column != 'PPC_all':
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='PPC_all', y=column, data=df)
            plt.title(f'Distribución según la presencia de infección: {column}')
            plt.xlabel('PPC_all (0 = No, 1 = Sí)')
            plt.ylabel(column)
            file_path = os.path.join(output_dir, f'{column}_boxplot_CI_1.png')
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            print(f"Gráfica guardada: {file_path}")

# Función para guardar resultados en Excel
def guardar_resultados_excel(desc_stats, output_file):
    """
    Guarda las estadísticas descriptivas en un archivo Excel.
    
    Args:
    - desc_stats (pd.DataFrame): Estadísticas descriptivas calculadas.
    - output_file (str): Ruta del archivo de salida.
    """
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            desc_stats.to_excel(writer, sheet_name='Estadísticas Descriptivas')
        print(f"Resultados guardados en: {output_file}")
    except Exception as e:
        print(f"Error al guardar resultados en Excel: {e}")

# Parámetros iniciales
file_path = 'iProve_gen_30.xlsx'  # Cambiar por la ruta correcta
output_dir = '/img'  # Carpeta para guardar gráficas
output_file = 'gen_desc_30.xlsx'  # Archivo Excel para guardar resultados

columns_of_interest = ['edad', 'genero', 'altura', 'peso', 'IMC', 'cirugiaAbdominalProgramada', 'cirugiaAbdominalUrgencia', 'esCirugiaOncologica', 'ASA', 'aRISCAT', 'spO2pre', 'hbPreoperatoria', 'infeccionRespiratoriaUltimoMes', 'hipertensionArterial', 'DM', 'cardiopatiaIsquemica', 'fumador', 'consumoAlcohol', 'dislipemia', 'ePOC', 'insuficienciaRenal', 'peepFinal', 'frFinal', 'vtFinal', 'Vt_ml_kg', 'spo2Final', 'fio2T3', 'pao2Final', 'paco2Final', 'presionMesetaFinal', 'dPressure', 'cristaloides', 'duracionCirugia', 'usoFarmacosVasoactivos', 'rever1onRNM', 'epidural', 'maniobraRescateIntraoperatorio', 'maniobraRescatePostoperatorio', 'Severe PPCs_0', 'Moderate PPCs_0', 'Severe PPCs_1', 'Moderate PPCs_1', 'Mild PPCS_1', 'Severe PPCs_2', 'Moderate PPCs_2', 'Severe PPCs_3', 'Moderate PPCs_3', 'Mild PPCS_3', 'Severe PPCs_5', 'Moderate PPCs_5', 'Mild PPCS_5', 'Severe PPCs_7', 'Moderate PPCs_7', 'Mild PPCS_7', 'Severe PPCs_30', 'Moderate PPCs_30', 'Severe_all', 'Severe_5', 'Moderate_all', 'Moderate_5', 'Mild_all', 'Mild_5', 'PPC_all', 'ID', 'Airtest', 'AirtestDico']
# Cargar datos
df_clean = cargar_datos(file_path, columns_of_interest)

if df_clean is not None:
    # Mostrar y guardar estadísticas descriptivas
    desc_stats = estadisticas_descriptivas(df_clean, columns_of_interest)
    guardar_resultados_excel(desc_stats, output_file)

    # Preguntar al usuario si desea visualizar distribuciones
#    check = input('¿Visualizar la distribución de los valores según CI y guardar gráficas? (Y/N): ').strip().upper()
#    if check == 'Y':
#        visualizar_distribuciones(df_clean, columns_of_interest, output_dir)
#    else:
#        print("Visualización de distribuciones omitida.")
else:
    print("No se pudo cargar el archivo. Revisa la ruta o formato del archivo.")
