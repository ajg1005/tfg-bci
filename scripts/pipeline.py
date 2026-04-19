from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


def get_models(random_state=42):
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced"
            )),
        ]),
         "rbf_svc": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=random_state
            )),
        ]),
        
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            )),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                criterion="entropy",
                max_features="sqrt",
                bootstrap=True,
                class_weight="balanced",
                random_state=random_state

            )),
        ]),

         "xgboost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state
            )),
        ]),
    }

    return models