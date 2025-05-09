import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Crear carpeta para guardar imágenes
output_dir = './img'
os.makedirs(output_dir, exist_ok=True)

# Ruta para guardar resultados en Excel
output_excel = 'iprove_clustering.xlsx'

# Limpieza y preprocesamiento
clustering_columns = ['edad', 'genero', 'altura', 'peso', 'IMC', 'cirugiaAbdominalProgramada', 'cirugiaAbdominalUrgencia', 'esCirugiaOncologica', 'ASA', 'aRISCAT', 'spO2pre', 'hbPreoperatoria', 'infeccionRespiratoriaUltimoMes', 'hipertensionArterial', 'DM', 'cardiopatiaIsquemica', 'fumador', 'consumoAlcohol', 'dislipemia', 'ePOC', 'insuficienciaRenal', 'peepFinal', 'frFinal', 'vtFinal', 'Vt_ml/kg', 'spo2Final', 'fio2T3', 'pao2Final', 'paco2Final', 'presionMesetaFinal', 'dPressure', 'cristaloides', 'duracionCirugia', 'usoFarmacosVasoactivos', 'rever1onRNM', 'epidural', 'maniobraRescateIntraoperatorio', 'maniobraRescatePostoperatorio', 'Severe PPCs_0', 'Moderate PPCs_0', 'Severe PPCs_1', 'Moderate PPCs_1', 'Mild PPCS_1', 'Severe PPCs_2', 'Moderate PPCs_2', 'Severe PPCs_3', 'Moderate PPCs_3', 'Mild PPCS_3', 'Severe PPCs_5', 'Moderate PPCs_5', 'Mild PPCS_5', 'Severe PPCs_7', 'Moderate PPCs_7', 'Mild PPCS_7', 'Severe PPCs_30', 'Moderate PPCs_30', 'Severe_all', 'Severe_5', 'Moderate_all', 'Moderate_5', 'Mild_all', 'Mild_5', 'PPC_all', 'ID', 'Airtest', 'AirtestDico']
df_clean = pd.read_excel('iProve_gen.xlsx').dropna(subset=clustering_columns, thresh=5)

# Imputar valores faltantes con la mediana
df_clean = df_clean[clustering_columns]
df_clean.fillna(df_clean.median(), inplace=True)

# Estandarizar las variables
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=clustering_columns)

# Determinar el número óptimo de clústeres
inertia = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Elegir K óptimo y aplicar clustering
k_optimal = 2
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10).fit(df_scaled)
df_clean['Cluster'] = kmeans.labels_
# Guardar gráfica del método del codo
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Método del Codo para determinar K')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Inercia')
plt.savefig(os.path.join(output_dir, 'codo.png'), bbox_inches='tight')
plt.close()

# Elegir K óptimo y aplicar clustering
k_optimal = 2
kmeans = KMeans(n_clusters=k_optimal, random_state=42).fit(df_scaled)
df_clean['Cluster'] = kmeans.labels_

# Guardar gráfico de clústeres con PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 5))
for cluster in range(k_optimal):
    cluster_points = pca_result[df_clean['Cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.title('Clustering de la Población (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.savefig(os.path.join(output_dir, 'clusters_pca.png'), bbox_inches='tight')
plt.close()

# Resultados de clustering descriptivo
cluster_summary = df_clean.groupby('Cluster').mean()

# Contrastes de hipótesis entre clústeres
contrast_results_clusters = []
for var in clustering_columns:
    cluster_groups = [df_clean[df_clean['Cluster'] == cluster][var].dropna() for cluster in range(k_optimal)]
    if all(len(group) > 1 for group in cluster_groups):
        try:
            stat, p_value = f_oneway(*cluster_groups)
            test_used = "ANOVA"
        except:
            stat, p_value = kruskal(*cluster_groups)
            test_used = "Kruskal-Wallis"
    else:
        stat, p_value, test_used = None, None, "No suficientes datos"
    contrast_results_clusters.append({"Variable": var, "Test usado": test_used, "Estadístico": stat, "p-valor": p_value})

contrast_results_clusters_df = pd.DataFrame(contrast_results_clusters)

# Guardar todos los resultados en Excel
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df_clean.to_excel(writer, sheet_name='Datos Clusterizados', index=False)
    cluster_summary.to_excel(writer, sheet_name='Resumen Clusters')
    contrast_results_clusters_df.to_excel(writer, sheet_name='Contrastes Clusters', index=False)

print(f"Resultados guardados en: {output_excel}")
print(f"Gráficas guardadas en la carpeta: {output_dir}")
