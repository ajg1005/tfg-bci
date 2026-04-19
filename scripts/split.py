from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def make_group_shuffle_split(
    X,
    y,
    groups,
    test_size=0.2,
    random_state=42,
):
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)

    subject_df = pd.DataFrame({
        "group": groups_array,
        "label": y_array,
    })

    # Cada sujeto debe tener una sola etiqueta
    label_counts = subject_df.groupby("group")["label"].nunique()
    inconsistent_groups = label_counts[label_counts != 1]
    if not inconsistent_groups.empty:
        raise ValueError(
            "Hay sujetos con más de una etiqueta y no se puede estratificar el split final."
        )

    # Una fila por sujeto para estratificar por clase a nivel de paciente
    subject_labels = (
        subject_df.groupby("group", sort=False)["label"]
        .first()
        .reset_index()
    )

    train_groups, test_groups = train_test_split(
        subject_labels["group"],
        test_size=test_size,
        random_state=random_state,
        stratify=subject_labels["label"],
    )

    train_idx = np.flatnonzero(np.isin(groups_array, train_groups.to_numpy()))
    test_idx = np.flatnonzero(np.isin(groups_array, test_groups.to_numpy()))

    if hasattr(X, "iloc"):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    if hasattr(y, "iloc"):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        y_train = y_array[train_idx]
        y_test = y_array[test_idx]

    if hasattr(groups, "iloc"):
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]
    else:
        groups_train = groups_array[train_idx]
        groups_test = groups_array[test_idx]

    return X_train, X_test, y_train, y_test, groups_train, groups_test


from sklearn.model_selection import GroupKFold

#stratified probar
def make_group_kfold_splits(
    X,
    y,
    groups,
    n_splits=5,
):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        # Separar X
        if hasattr(X, "iloc"):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]

        # Separar y
        if hasattr(y, "iloc"):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            y_train = y[train_idx]
            y_test = y[test_idx]

        # Separar groups
        if hasattr(groups, "iloc"):
            groups_train = groups.iloc[train_idx]
            groups_test = groups.iloc[test_idx]
        else:
            groups_train = groups[train_idx]
            groups_test = groups[test_idx]

        splits.append({
            "fold": fold,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "groups_train": groups_train,
            "groups_test": groups_test,
        })

    return splits
#LOSO Leave one out