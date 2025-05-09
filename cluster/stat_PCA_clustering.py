#!/usr/bin/env python3
"""
Script para identificar las variables que más diferencian dos clusters
y guardar los resultados en un fichero Excel con pestañas separadas:
 - Loadings
 - Top_PC1
 - Top_PC2
 - Top_Diff
 - TTest
Uso:
    python3 stat_PCA_clustering.py entrada.xlsx [salida.xlsx]
"""

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind


def main(path_excel: str, output_excel: str):
    # 1) Cargar datos
    df = pd.read_excel(path_excel)
    if 'Cluster' not in df.columns:
        raise ValueError("El fichero debe contener una columna 'Cluster' con valores 0/1.")
    
    # 2) Selección de variables (excluir ID y Cluster)
    exclude = ['ID', 'Cluster']
    features = [c for c in df.columns if c not in exclude]
    
    # 3) Estandarizar
    X = df[features].fillna(df[features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4) PCA a 2 componentes
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    
    # 5) Calcular loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=['PC1', 'PC2']
    )
    
    # 6) Top 5 variables en cada componente
    top_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(5)
    top_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(5)
    
    # 7) Diferencia de medias por cluster
    means = df.groupby('Cluster')[features].mean().T
    means['abs_diff'] = (means[0] - means[1]).abs()
    top_diff = means.sort_values('abs_diff', ascending=False).head(5)
    
    # 8) t-test para Top_Diff
    ttest_results = []
    for var in top_diff.index:
        g0 = df[df.Cluster == 0][var].dropna()
        g1 = df[df.Cluster == 1][var].dropna()
        stat, p = ttest_ind(g0, g1, equal_var=False)
        ttest_results.append({'Variable': var, 't_stat': stat, 'p_value': p})
    ttest_df = pd.DataFrame(ttest_results)
    
    # 9) Exportar a Excel
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        loadings.to_excel(writer, sheet_name='Loadings')
        top_pc1.to_frame('abs_loading').to_excel(writer, sheet_name='Top_PC1')
        top_pc2.to_frame('abs_loading').to_excel(writer, sheet_name='Top_PC2')
        top_diff[['abs_diff']].to_excel(writer, sheet_name='Top_Diff')
        ttest_df.to_excel(writer, sheet_name='TTest', index=False)

    print(f"Resultados guardados en '{output_excel}'")

if __name__ == '__main__':
    excel = "iprove_clustering.xlsx"
    output = "iprove_clustering_PCA.xlsx"
    main(excel, output)
