import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_bootstrap_ci(y_true, x_score, n_bootstraps=1000, alpha=0.05, random_state=None):
    """
    Calcular intervalos de confianza mediante bootstrap para AUC, sensibilidad y especificidad al umbral de Youden.
    """
    rng = np.random.RandomState(random_state)
    aucs = []
    sens = []
    specs = []
    thrs = []
    n = len(y_true)

    # Pre-cálculo del punto de referencia para Youden en la muestra original
    fpr0, tpr0, thr0 = roc_curve(y_true, x_score)
    youden0 = tpr0 - fpr0
    idx0 = np.nanargmax(youden0)
    base_thr = thr0[idx0]

    for i in range(n_bootstraps):
        # bootstrap sample indices
        idxs = rng.randint(0, n, n)
        y_bs = y_true[idxs]
        x_bs = x_score[idxs]

        # evitar casos sin ambas clases
        if len(np.unique(y_bs)) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y_bs, x_bs)
        roc_auc = auc(fpr, tpr)

        youden = tpr - fpr
        j = np.nanargmax(youden)
        thr_bs = thresholds[j]
        sens_bs = tpr[j]
        spec_bs = 1 - fpr[j]

        aucs.append(roc_auc)
        sens.append(sens_bs)
        specs.append(spec_bs)
        thrs.append(thr_bs)

    # calcular percentiles
    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)
    ci = {
        'AUC_CI_low': np.percentile(aucs, lower_p),
        'AUC_CI_high': np.percentile(aucs, upper_p),
        'Sensitivity_CI_low': np.percentile(sens, lower_p),
        'Sensitivity_CI_high': np.percentile(sens, upper_p),
        'Specificity_CI_low': np.percentile(specs, lower_p),
        'Specificity_CI_high': np.percentile(specs, upper_p),
        'Threshold_CI_low': np.percentile(thrs, lower_p),
        'Threshold_CI_high': np.percentile(thrs, upper_p),
    }
    return ci


def main(input_excel, output_excel, n_bootstraps=1000):
    # Leer datos
    df = pd.read_excel(input_excel)
    if 'PPC_all' not in df.columns:
        raise KeyError("La columna 'Severe_all' no está en el fichero.")

    # Directorio para gráficos
    out_dir = os.path.join('img', 'roc_plots')
    os.makedirs(out_dir, exist_ok=True)

    # Variable objetivo
    y = (df['PPC_all'] == 1).astype(int).values

    # Variables numéricas a evaluar
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c != 'PPC_all']

    # Lista para resultados
    results = []

    # Procesar cada predictor
    for var in predictors:
        x = df[var]
        mask = x.notna().values
        y_valid = y[mask]
        x_valid = x[mask].astype(float).values

        # Calcular ROC/AUC
        fpr, tpr, thresholds = roc_curve(y_valid, x_valid)
        roc_auc = auc(fpr, tpr)

        # Encontrar umbral óptimo por índice de Youden
        youden = tpr - fpr
        idx = np.nanargmax(youden)
        best_thr = thresholds[idx]
        best_sens = tpr[idx]
        best_spec = 1 - fpr[idx]

        # Calcular CI de bootstrap
        ci = compute_bootstrap_ci(y_valid, x_valid, n_bootstraps=n_bootstraps, alpha=0.05, random_state=42)

        # Guardar resultados
        results.append({
            'Variable': var,
            'AUC': roc_auc,
            'AUC_CI_low': ci['AUC_CI_low'],
            'AUC_CI_high': ci['AUC_CI_high'],
            'BestThreshold': best_thr,
            'Threshold_CI_low': ci['Threshold_CI_low'],
            'Threshold_CI_high': ci['Threshold_CI_high'],
            'Sensitivity': best_sens,
            'Sensitivity_CI_low': ci['Sensitivity_CI_low'],
            'Sensitivity_CI_high': ci['Sensitivity_CI_high'],
            'Specificity': best_spec,
            'Specificity_CI_low': ci['Specificity_CI_low'],
            'Specificity_CI_high': ci['Specificity_CI_high'],
        })

        # Graficar ROC
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f} (95% CI [{ci['AUC_CI_low']:.3f}, {ci['AUC_CI_high']:.3f}])")
        plt.scatter(fpr[idx], tpr[idx], color='red',
                    label=f'Youden@{best_thr:.2f}\nSens={best_sens:.2f}, Spec={best_spec:.2f}')
        plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: {var}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'ROC_{var}_ppc_all_val.png'))
        plt.close()

    # Exportar métricas a Excel
    df_out = pd.DataFrame(results)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='Metrics')

    print(f"Resultados guardados en '{output_excel}'.")
    print(f"Gráficos en '{out_dir}/'.")


if __name__ == '__main__':
    # Parámetros de entrada
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'iProve_gen_30.xlsx'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'iProve_rocs_multiples_ppc_ci.xlsx'
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    main(input_file, output_file, n_bootstraps=n_boot)
