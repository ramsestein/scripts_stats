#!/usr/bin/env python3
"""
normalizar_excel.py

Normaliza todas las columnas numéricas de un Excel al rango [0, 1]
y guarda el resultado en otro Excel.

Edita las variables INPUT_FILE y OUTPUT_FILE según tus necesidades.
"""

from pathlib import Path
import pandas as pd

# ⇩⇩⇩  MODIFICA AQUÍ LOS NOMBRES DE ARCHIVO  ⇩⇩⇩
INPUT_FILE  = Path("iprove.xlsx")        # Excel original
OUTPUT_FILE = Path("iprove_n.xlsx") # Excel normalizado
# ⇧⇧⇧  MODIFICA AQUÍ LOS NOMBRES DE ARCHIVO  ⇧⇧⇧


def normalizar_excel(in_path: Path, out_path: Path) -> None:
    """Lee in_path, normaliza columnas numéricas y escribe out_path."""
    # 1) Leer la primera hoja (cambia sheet_name=... si lo necesitas)
    df = pd.read_excel(in_path)

    # 2) Seleccionar columnas numéricas
    num_cols = df.select_dtypes(include="number").columns

    # 3) Calcular min y max
    col_min = df[num_cols].min()
    col_max = df[num_cols].max()

    # 4) Normalizar evitando división por cero
    rango = col_max - col_min
    rango[rango == 0] = 1          # columnas constantes → todo 0
    df[num_cols] = (df[num_cols] - col_min) / rango

    # 5) Guardar
    df.to_excel(out_path, index=False)
    print(f"Archivo normalizado guardado en → {out_path.resolve()}")


if __name__ == "__main__":
    normalizar_excel(INPUT_FILE, OUTPUT_FILE)
