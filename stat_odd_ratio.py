import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import Table2x2

# 1) Asegúrate de tener instalados:
#    pip install pandas statsmodels openpyxl

# Parámetros
INPUT_FILE      = 'iProve_gen_30.xlsx'
OUTPUT_FILE     = 'odds_ratios_bootstrap.xlsx'
N_BOOTSTRAP     = 1000
RANDOM_SEED     = 42

# Semilla para reproducibilidad
np.random.seed(RANDOM_SEED)

# Carga de datos
df = pd.read_excel(INPUT_FILE)

# Definir exposiciones y desenlace
exposures = [
    'spO2pre_AND_AirTest',
    'spO2pre_OR_AirTest',
    'AirTestpre AND AirTestpost AND',
    'AirTestpre AND AirTestpost OR'
]
outcome = 'Severe_all'

# Comprobar columnas
missing = [c for c in exposures + [outcome] if c not in df.columns]
if missing:
    raise KeyError(f"No encontré estas columnas en {INPUT_FILE}: {missing}")

results = []

for exp in exposures:
    # Contingencia en la muestra original
    ct = pd.crosstab(df[exp], df[outcome])
    a = ct.at[1,1] if (1 in ct.index and 1 in ct.columns) else 0
    b = ct.at[1,0] if (1 in ct.index and 0 in ct.columns) else 0
    c = ct.at[0,1] if (0 in ct.index and 1 in ct.columns) else 0
    d = ct.at[0,0] if (0 in ct.index and 0 in ct.columns) else 0

    # OR puntual asintótico
    table     = Table2x2([[a, b], [c, d]])
    or_est    = table.oddsratio
    p_val     = table.oddsratio_pvalue()

    # Bootstrap para estimar IC del OR
    boot_ors = []
    for i in range(N_BOOTSTRAP):
        df_boot = df.sample(n=len(df), replace=True)
        ctb = pd.crosstab(df_boot[exp], df_boot[outcome])
        a_b = c_b = d_b = b_b = 0
        a_b = ctb.at[1,1] if (1 in ctb.index and 1 in ctb.columns) else 0
        b_b = ctb.at[1,0] if (1 in ctb.index and 0 in ctb.columns) else 0
        c_b = ctb.at[0,1] if (0 in ctb.index and 1 in ctb.columns) else 0
        d_b = ctb.at[0,0] if (0 in ctb.index and 0 in ctb.columns) else 0
        # Corrección de Haldane–Anscombe si hay ceros
        if min(a_b, b_b, c_b, d_b) == 0:
            a_b += 0.5; b_b += 0.5; c_b += 0.5; d_b += 0.5
        boot_ors.append((a_b * d_b) / (b_b * c_b))

    ci_low, ci_upp = np.percentile(boot_ors, [2.5, 97.5])

    results.append({
        'Variable':            exp,
        'OR':                  round(or_est, 2),
        'CI95%_low_boot':      round(ci_low, 2),
        'CI95%_up_boot':       round(ci_upp, 2),
        'p-value_asintótico':  round(p_val, 4),
        'N_expuestos':         int(a + b),
        'N_no_expuestos':      int(c + d)
    })

# Volcar resultados a Excel
res_df = pd.DataFrame(results)
res_df.to_excel(OUTPUT_FILE, index=False)

print(f"Resultados con bootstrap guardados en: {OUTPUT_FILE}\n")
print(res_df.to_string(index=False))
