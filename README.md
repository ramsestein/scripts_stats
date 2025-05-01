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
### Preprocesamiento de datos

#### stat_eliminar_missing.py
Elimina filas con valores nulos en SpO2_1 y SpO2_2.

#### stat_crear_nueva_variable.py
Añade la columna Saturation_Lowest (mínimo entre SpO2_1 y SpO2_2).

#### stat_sustraccion_poblacion.py
Resta dos datasets basados en la clave Study+Patient.

#### stat_separar_aleatorio.py
Divide aleatoriamente en un 30 % y un 70 %.

#### stat_separar_subgrupos.py
Separa en dos grupos según spO2pre (< 97 vs ≥ 97).

### Estadística descriptiva y correlación

#### stat_descriptivo.py
Limpia datos, calcula descripciones y genera boxplots opcionales.

#### stat_correlacion.py
Calcula matriz de correlación y muestra un mapa de calor.

### Pruebas de hipótesis

#### stat_normalidad_y_contrastes.py
Shapiro–Wilk + t-test/ANOVA o Mann–Whitney/Kruskal–Wallis según normalidad.

#### stat_contraste_3_excels.py
Compara dos Excel con pruebas de normalidad y test adecuados.

#### stat_contraste_3_excels_confusores.py
Igual que el anterior, ajustando por confusores via ANCOVA y bootstrap.

#### stat_odd_ratio.py
Calcula odds ratios, IC y p-valores para exposiciones vs desenlace binario.

### Modelado y análisis predictivo

#### stat_regresion_lineal.py
Ajusta regresiones lineales simples y exporta coeficientes.

#### stat_GEE.py
Modelos GEE para desenlace binario con confusores.

#### stat_predictivos.py
Regresión logística, Random Forest, XGBoost y redes neuronales con validación cruzada.

### Análisis multivariante y clustering

#### stat_PCA_clustering.py
PCA a 2 componentes, selección de variables clave y reportes.

#### stat_clustering.py
K-Means con “método del codo”, visualización y contrastes entre clusters.

### Evaluación de clasificadores y diagnóstico

#### stat_roc_univariante.py
Curvas ROC univariantes, AUC e IC por bootstrap.

## Contribuciones
¡Las contribuciones son bienvenidas! Abre issues o pull requests para mejorar o ampliar los scripts.

## Licencia
Este proyecto está bajo la licencia MIT.
