import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, shapiro

def realizar_comparaciones(df1, df2, columnas):
    resultados = []
    columnas_no_numericas = []

    for columna in columnas:
        # Verifica si la columna es numérica en ambos DataFrames
        if not (pd.api.types.is_numeric_dtype(df1[columna]) and pd.api.types.is_numeric_dtype(df2[columna])):
            columnas_no_numericas.append(columna)
            continue

        datos1 = df1[columna].dropna()
        datos2 = df2[columna].dropna()

        # Comprueba que haya datos suficientes
        if len(datos1) < 3 or len(datos2) < 3:
            continue  # Saltar si no hay suficientes datos para análisis

        # Test de normalidad Shapiro-Wilk
        _, p_normalidad1 = shapiro(datos1)
        _, p_normalidad2 = shapiro(datos2)

        if p_normalidad1 > 0.05 and p_normalidad2 > 0.05:
            # Datos normales: T-Test
            stat, p_valor = ttest_ind(datos1, datos2, equal_var=False)
            resultados.append({
                "Variable": columna,
                "Estadístico": stat,
                "p-valor": p_valor,
                "Prueba": "T-Test"
            })
        else:
            # Datos no normales: Mann-Whitney U
            stat_mw, p_valor_mw = mannwhitneyu(datos1, datos2, alternative='two-sided')

            # Preparación para Chi-cuadrado (discretización en cuartiles)
            combinados = pd.concat([datos1, datos2]).reset_index(drop=True)
            try:
                discretizados = pd.qcut(combinados, q=4, duplicates='drop')
                grupos = pd.Series(['Grupo1']*len(datos1) + ['Grupo2']*len(datos2))

                # Tabla de contingencia
                tabla_contingencia = pd.crosstab(grupos, discretizados)

                # Chi-cuadrado
                stat_chi2, p_valor_chi2, _, _ = chi2_contingency(tabla_contingencia)

                resultados.append({
                    "Variable": columna,
                    "Estadístico MW": stat_mw,
                    "p-valor MW": p_valor_mw,
                    "Estadístico Chi2": stat_chi2,
                    "p-valor Chi2": p_valor_chi2,
                    "Prueba": "Mann-Whitney U y Chi-cuadrado"
                })

            except ValueError as e:
                print(f"Error en Chi-cuadrado para columna '{columna}': {e}")
                resultados.append({
                    "Variable": columna,
                    "Estadístico MW": stat_mw,
                    "p-valor MW": p_valor_mw,
                    "Estadístico Chi2": np.nan,
                    "p-valor Chi2": np.nan,
                    "Prueba": "Mann-Whitney U (Chi-cuadrado falló)"
                })

    return resultados, columnas_no_numericas

# Ejemplo de uso
data1 = pd.read_excel('iProve_gen_30.xlsx')
data2 = pd.read_excel('iProve_gen_70.xlsx')
columnas = data1.columns.intersection(data2.columns)

resultados, columnas_no_numericas = realizar_comparaciones(data1, data2, columnas)

# Guardar resultados en Excel
resultados_df = pd.DataFrame(resultados)
resultados_df.to_excel("iProve_gen_comparacion.xlsx", index=False)

# Imprimir resultados y columnas no numéricas
print("Resultados de comparaciones estadísticas:\n", resultados_df)
print("\nColumnas no numéricas excluidas:", columnas_no_numericas)

