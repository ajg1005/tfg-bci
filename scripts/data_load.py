from pathlib import Path
import pandas as pd
# Cargar el dataset 
def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("El dataset está vacío")
    
    return df

