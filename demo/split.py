from sklearn.model_selection import GroupShuffleSplit


def make_group_shuffle_split(
    X,
    y,
    groups,
    test_size=0.2,
    random_state=42,
):
    # Crear la partición por grupos
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    # Obtener índices de train y test
    train_idx, test_idx = next(splitter.split(X, y, groups))

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

    return X_train, X_test, y_train, y_test, groups_train, groups_test

from sklearn.model_selection import GroupKFold


def make_group_kfold_splits(
    X,
    y,
    groups,
    n_splits=5,
):
    splitter = GroupKFold(n_splits=n_splits)
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