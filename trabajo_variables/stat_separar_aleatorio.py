
import sys
import pandas as pd

def main(input_path: str, out30: str, out70: str, frac: float = 0.3, random_state: int = 42):
    # 1) Carga del Excel original
    df = pd.read_excel(input_path)
    
    # 2) Muestra aleatoria del 30 %
    df_30 = df.sample(frac=frac, random_state=random_state)
    
    # 3) Resto del 70 %
    df_70 = df.drop(df_30.index)
    
    # 4) Guardar ambos
    df_30.to_excel(out30, index=False)
    df_70.to_excel(out70, index=False)
    
    print(f"Guardado {len(df_30)} filas en {out30} ({frac*100:.0f}%)")
    print(f"Guardado {len(df_70)} filas en {out70} ({(1-frac)*100:.0f}%)")

if __name__ == "__main__":
    inp = "iProve_gen.xlsx"
    out1 = "iProve_gen_30.xlsx"
    out2 = "iProve_gen_70.xlsx"
    main(inp, out1, out2, frac=0.3)
