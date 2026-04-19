from pathlib import Path

from data_load import load_dataset
from preprocessing import inspect_dataset, preprocess_dataset
from epochs import create_epochs


csv_path = Path(__file__).resolve().parent.parent / "data" / "adhdata.csv"

df = load_dataset(csv_path)

print("Dataset original")
inspect_dataset(df)

df_clean, eeg_cols = preprocess_dataset(df)

print("\nDataset preprocesado")
inspect_dataset(df_clean)
print("\nColumnas EEG:", eeg_cols)

X_epochs, y_epochs, groups_epochs = create_epochs(
    df=df_clean,
    eeg_columns=eeg_cols,
    label_column="Class",
    group_column="ID",
    epoch_size=128,
    step_size=64,
)

print("\nShape X_epochs:", X_epochs.shape)
print("Shape y_epochs:", y_epochs.shape)
print("Shape groups_epochs:", groups_epochs.shape)

