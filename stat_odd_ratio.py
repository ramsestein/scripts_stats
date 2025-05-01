import pandas as pd
from statsmodels.stats.contingency_tables import Table2x2

# 1) Asegúrate de tener instalados:
#    pip install pandas statsmodels openpyxl

# 2) Ruta a tu fichero de datos
INPUT_FILE = 'iProve_gen_70.xlsx'
OUTPUT_FILE = 'odds_ratios.xlsx'

# 3) Carga de datos
df = pd.read_excel(INPUT_FILE)

# 2. Definir exposiciones y desenlace
exposures = ['spO2pre_AND_AirTest', 'spO2pre_OR_AirTest', 'AirTestpre AND AirTestpost AND', 'AirTestpre AND AirTestpost OR']
outcome = 'Severe_all'

# 5) Comprueba que existan las columnas
missing = [c for c in exposures + [outcome] if c not in df.columns]
if missing:
    raise KeyError(f"No encontré estas columnas en {INPUT_FILE}: {missing}")

# 6) Calcula OR para cada exposición
results = []
for exp in exposures:
    ct = pd.crosstab(df[exp], df[outcome])
    a = ct.at[1,1] if (1 in ct.index and 1 in ct.columns) else 0
    b = ct.at[1,0] if (1 in ct.index and 0 in ct.columns) else 0
    c = ct.at[0,1] if (0 in ct.index and 1 in ct.columns) else 0
    d = ct.at[0,0] if (0 in ct.index and 0 in ct.columns) else 0

    table     = Table2x2([[a, b], [c, d]])
    or_est    = table.oddsratio
    ci_low, ci_upp = table.oddsratio_confint()
    p_val     = table.oddsratio_pvalue()

    results.append({
        'Variable': exp,
        'OR': round(or_est, 2),
        'CI95%_low': round(ci_low, 2),
        'CI95%_up':  round(ci_upp, 2),
        'p-value': round(p_val, 4),
        'N_expuestos': a + b,
        'N_no_expuestos': c + d
    })

# 7) Volcar a Excel
res_df = pd.DataFrame(results)
res_df.to_excel(OUTPUT_FILE, index=False)

# 8) Informar por consola
print(f"✅ Resultados guardados en: {OUTPUT_FILE}\n")
print(res_df.to_string(index=False))
