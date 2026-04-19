import pandas as pd



# Mirar informacion basica del dataset
def inspect_dataset(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nTipos:")
    print(df.dtypes)

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nPrimeras filas:")
    print(df.head().to_string())

    if "Class" in df.columns:
        print("\nDistribución de clases:")
        print(df["Class"].value_counts(dropna=False))

    if "ID" in df.columns:
        print("\nNúmero de pacientes únicos:")
        print(df["ID"].nunique())


def preprocess_dataset(
    df: pd.DataFrame,
    subject_col: str = "ID",
    label_col: str = "Class"
) -> tuple[pd.DataFrame, list[str]]:
    # Copia
    df = df.copy()

    # Comprobar columnas
    required_cols = [subject_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # Eliminar filas sin identificador de sujeto o sin etiqueta
    df = df.dropna(subset=[subject_col, label_col])

    # Codificar las clases
    label_map = {"Control": 0, "ADHD": 1}
    df[label_col] = df[label_col].map(label_map)


    # Variables EEG, todas menos class e id
    eeg_cols = [col for col in df.columns if col not in [subject_col, label_col]]
    if not eeg_cols:
        raise ValueError("No se encontraron columnas EEG.")

    return df, eeg_cols