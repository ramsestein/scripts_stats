# 📊 scripts_stats

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)  
Repositorio de scripts estadísticos y de machine learning diseñados para facilitar el análisis exploratorio, la preparación de datos y la evaluación de modelos, especialmente en contextos clínicos o de investigación aplicada.

---

## Índice

- [Estructura del repositorio](#estructura-del-repositorio)
  - [analisis_confusor/](#analisis_confusor)
  - [basics/](#basics)
  - [cluster/](#cluster)
  - [eval_model/](#eval_model)
  - [trabajo_excel/](#trabajo_excel)
  - [trabajo_variables/](#trabajo_variables)
  - [train_model/](#train_model)
- [Uso](#uso)
- [Requisitos](#requisitos)
- [Contribuciones](#contribuciones)

---

## Estructura del repositorio

### `analisis_confusor/`
Scripts destinados a evaluar el impacto de variables confusoras sobre resultados clínicos o modelos predictivos.  
Incluye técnicas como regresión ajustada, comparación de modelos con y sin confusores, y visualizaciones.

### `basics/`
Contiene análisis estadístico básico: medias, medianas, desviaciones, pruebas de normalidad, t-test, U-Mann Whitney, chi-cuadrado, etc.  
Útil para análisis exploratorios y generación de tablas de resumen.

### `cluster/`
Implementaciones sencillas de algoritmos de clustering como K-means o DBSCAN, así como evaluación de la calidad de los agrupamientos.  
Incluye visualización de clústeres y análisis de perfiles de grupos.

### `eval_model/`
Evaluación de modelos supervisados (regresión, clasificación) mediante métricas como ROC, AUC, matriz de confusión, validación cruzada y curvas de calibración.

### `trabajo_excel/`
Automatización de tareas frecuentes con archivos `.xlsx`, como la carga múltiple de hojas, formateo, exportación de tablas y actualización de celdas desde scripts.

### `trabajo_variables/`
Scripts para transformar variables: recodificación de categóricas, escalado, binarización, creación de variables derivadas o análisis de correlación.

### `train_model/`
Entrenamiento de modelos predictivos incluyendo regresión logística, random forest, XGBoost y otros.  
Incluye técnicas de selección de variables, balanceo y guardado de modelos entrenados.

---

## Uso

Cada script se ejecuta directamente sin necesidad de argumentos por consola.  
Solo necesitas editar las variables indicadas al inicio (y a veces al final) del archivo para adaptarlo a tus datos o necesidades. Luego ejecuta el script con Python.

```bash
python nombre_del_script.py
````

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

## Contribuciones
¡Las contribuciones son bienvenidas! Abre issues o pull requests para mejorar o ampliar los scripts.

## Licencia
Este proyecto está bajo la licencia MIT.
