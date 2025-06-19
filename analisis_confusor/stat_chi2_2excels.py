import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

def compare_excels_chi2(
    path_a: str,
    path_b: str,
    sheet_a: str | int = 0,
    sheet_b: str | int = 0,
    min_exp: float = 5.0,          # umbral para decidir Fisher
    alpha: float = 0.05            # nivel de significación para resaltar
) -> pd.DataFrame:
    """
    Compara dos Excel columna por columna (0/1) con χ² de Pearson.
    
    Parameters
    ----------
    path_a, path_b : str
        Rutas a los dos ficheros Excel.
    sheet_a, sheet_b : str or int
        Nombre o índice de la hoja a leer en cada Excel.
    min_exp : float
        Si alguna frecuencia esperada es menor que este valor
        se usa Fisher exacto en vez de χ².
    alpha : float
        Nivel de significación para marcar resultados (*).
    
    Returns
    -------
    pandas.DataFrame
        Tabla con recuentos, χ², p-valor y método usado.
    """
    # 1. Leer los Excels
    df_a = pd.read_excel(path_a, sheet_name=sheet_a)
    df_b = pd.read_excel(path_b, sheet_name=sheet_b)

    # 2. Intersección de columnas
    common_cols = set(df_a.columns).intersection(df_b.columns)

    results = []

    for col in common_cols:
        # 3. Comprobar que la columna es binaria (0/1) en ambos archivos
        vals_a = df_a[col].dropna().unique()
        vals_b = df_b[col].dropna().unique()
        valid_vals = {0, 1}

        if set(vals_a).issubset(valid_vals) and set(vals_b).issubset(valid_vals):
            # 4. Construir tabla de contingencia 2×2
            a_zeros = (df_a[col] == 0).sum()
            a_ones  = (df_a[col] == 1).sum()
            b_zeros = (df_b[col] == 0).sum()
            b_ones  = (df_b[col] == 1).sum()

            table = [[a_zeros, a_ones],
                     [b_zeros, b_ones]]

            # 5. χ² de Pearson
            chi2, p, dof, exp = chi2_contingency(table, correction=False)
            method = "χ² Pearson"

            # Si hay alguna frecuencia esperada < min_exp ⇒ Fisher
            if exp.min() < min_exp:
                _, p = fisher_exact(table)
                chi2 = float("nan")          # Fisher no devuelve χ²
                method = "Fisher exacto"

            results.append({
                "Variable": col,
                "A_0": a_zeros, "A_1": a_ones,
                "B_0": b_zeros, "B_1": b_ones,
                "χ²": chi2,
                "p": p,
                "Método": method,
                "Significativo": "*" if p < alpha else ""
            })

    return pd.DataFrame(results)\
             .sort_values("p")

# --- Uso ----------------------------------------------------
if __name__ == "__main__":
    table = compare_excels_chi2(
        "iProve_genPPC_0.xlsx",
        "iProve_genPPC_1.xlsx",
        sheet_a=0,          # o "Descriptivo_study_population"
        sheet_b=0           # o "Descriptivo_validation"
    )
    print(table.to_string(index=False))
    # Si lo prefieres en Excel:
    table.to_excel("comparativa_chi2.xlsx", index=False)
