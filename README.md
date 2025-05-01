# scripts_stats
# scripts_stats

Una colección de scripts en Python para análisis estadístico completo: preprocesamiento, estadística descriptiva, pruebas de hipótesis, modelado, clustering y visualización.

---

## Requisitos

- **Python 3.7+**  
- **Librerías principales**:
  - `pandas`
  - `numpy`
  - `scipy`
  - `statsmodels`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `openpyxl`
  - `xgboost` (solo para `stat_predictivos.py`)

Para instalar todas las dependencias:
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn openpyxl xgboost

```

## Uso
Ejecuta cada script desde línea de comandos o impórtalo en tu proyecto:

```bash
python nombre_del_script.py --input datos.xlsx --output resultados.xlsx
```

## Descripción de los scripts
Preprocesamiento de datos

stat_eliminar_missing.py
Elimina filas con valores nulos en SpO2_1 y SpO2_2.

stat_crear_nueva_variable.py
Añade la columna Saturation_Lowest (mínimo entre SpO2_1 y SpO2_2).

stat_sustraccion_poblacion.py
Resta dos datasets basados en la clave Study+Patient.

stat_separar_aleatorio.py
Divide aleatoriamente en un 30 % y un 70 %.

stat_separar_subgrupos.py
Separa en dos grupos según spO2pre (< 97 vs ≥ 97).

Estadística descriptiva y correlación

stat_descriptivo.py
Limpia datos, calcula descripciones y genera boxplots opcionales.

stat_correlacion.py
Calcula matriz de correlación y muestra un mapa de calor.

Pruebas de hipótesis

stat_normalidad_y_contrastes.py
Shapiro–Wilk + t-test/ANOVA o Mann–Whitney/Kruskal–Wallis según normalidad.

stat_contraste_3_excels.py
Compara dos Excel con pruebas de normalidad y test adecuados.

stat_contraste_3_excels_confusores.py
Igual que el anterior, ajustando por confusores via ANCOVA y bootstrap.

stat_odd_ratio.py
Calcula odds ratios, IC y p-valores para exposiciones vs desenlace binario.

Modelado y análisis predictivo

stat_regresion_lineal.py
Ajusta regresiones lineales simples y exporta coeficientes.

stat_GEE.py
Modelos GEE para desenlace binario con confusores.

stat_predictivos.py
Regresión logística, Random Forest, XGBoost y redes neuronales con validación cruzada.

Análisis multivariante y clustering

stat_PCA_clustering.py
PCA a 2 componentes, selección de variables clave y reportes.

stat_clustering.py
K-Means con “método del codo”, visualización y contrastes entre clusters.

Evaluación de clasificadores y diagnóstico

stat_roc_univariante.py
Curvas ROC univariantes, AUC e IC por bootstrap.

Contribuciones
¡Las contribuciones son bienvenidas! Abre issues o pull requests para mejorar o ampliar los scripts.

Licencia
Este proyecto está bajo la licencia MIT.









Uso
Cada script está pensado para ejecutarse desde línea de comandos o importarse en otro proyecto. La mayoría acepta rutas de archivos de entrada y salida como argumentos. Consulta el encabezado de cada archivo para detalles de uso.

Scripts
Preprocesamiento de datos
stat_eliminar_missing.py
Filtra un archivo Excel para conservar solo las filas que no tienen valores nulos en las columnas SpO2_1 y SpO2_2. 
GitHub

stat_crear_nueva_variable.py
Añade una columna Saturation_Lowest al Excel de entrada, que corresponde al mínimo entre SpO2_1 y SpO2_2. 
GitHub

stat_sustraccion_poblacion.py
Resta dos poblaciones basándose en la clave Study+Patient, eliminando del primer fichero aquellas filas presentes en el segundo. 
GitHub

stat_separar_aleatorio.py
Separa aleatoriamente un dataset en un 30 % y un 70 %, guardando dos archivos Excel resultantes. 
GitHub

stat_separar_subgrupos.py
Divide los datos en dos subgrupos según un umbral de spO2pre (< 97 vs ≥ 97) y exporta cada grupo a Excel. 
GitHub

Estadística descriptiva y correlación
stat_descriptivo.py
Carga datos desde Excel, limpia valores nulos, calcula estadísticas descriptivas y exporta resultados; opcionalmente genera boxplots. 
GitHub

stat_correlacion.py
Calcula la matriz de correlación de variables numéricas y genera un mapa de calor, guardando ambos en disco. 
GitHub

Pruebas de hipótesis
stat_normalidad_y_contrastes.py
Verifica la normalidad de variables con Shapiro-Wilk y, según esto, aplica t-test, ANOVA o pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis). 
GitHub

stat_contraste_3_excels.py
Compara dos ficheros Excel mediante pruebas de normalidad y elige entre t-test, Mann-Whitney U y chi-cuadrado tras discretizar. 
GitHub

stat_contraste_3_excels_confusores.py
Extiende lo anterior ajustando además por múltiples confusores con ANCOVA y bootstrap para estimar diferencias ajustadas. 
GitHub

stat_odd_ratio.py
Calcula odds ratios, intervalos de confianza y p-valores para varias exposiciones frente a un desenlace binario. 
GitHub

Modelado y análisis predictivo
stat_regresion_lineal.py
Ajusta regresiones lineales simples de SpO2_1 y SpO2_2 frente a airTest_SpO2, exportando coeficientes y estadísticas a Excel. 
GitHub

stat_GEE.py
Implementa modelos de Ecuaciones de Estimación Generalizada (GEE) para un desenlace binario (Composite infección), ajustando confusores y guardando resultados. 
GitHub

stat_predictivos.py
Entrena y evalúa modelos predictivos (Regresión logística, Random Forest, XGBoost, Redes neuronales) con imputación por subgrupos y validación cruzada. 
GitHub

Análisis multivariante y clustering
stat_PCA_clustering.py
Realiza PCA para reducir dimensiones a dos componentes, identifica variables clave (loadings, diferencias de medias) y exporta varios reportes a Excel. 
GitHub

stat_clustering.py
Estandariza datos, determina K óptimo (“método del codo”), aplica K-Means, visualiza con PCA y realiza contrastes de hipótesis entre clústeres. 
GitHub

Evaluación de clasificadores y diagnóstico
stat_roc_univariante.py
Calcula curvas ROC univariantes para cada predictor numérico vs un desenlace binario, obtiene AUC y sus IC por bootstrap, y guarda gráficos. 
GitHub

Contribuciones
Siéntete libre de abrir issues o pull requests para mejorar los scripts, añadir nuevas funcionalidades o corregir errores.

Licencia
Este repositorio está disponible bajo la licencia MIT.
