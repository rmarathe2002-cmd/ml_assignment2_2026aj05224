"""Train 5 classifiers on ANY csv (numeric + categorical columns, missing values ok).

Usage:
  python train_models.py --data mydata.csv                    # target = last column
  python train_models.py --data mydata.csv --target Churn --drop CustomerID,Name
Enforces the assignment minimums: >= 12 features and >= 500 instances.
"""
import argparse
import json
import sys
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics, MIN_FEATURES, MIN_INSTANCES

warnings.filterwarnings("ignore", category=FutureWarning)
MAX_CLASSES = 20


def fail(msg):
    print(f"\n[REJECTED] {msg}")
    sys.exit(1)


def prepare(df, target, drop):
    if target is None:
        target = df.columns[-1]
        print(f"No --target given, using last column: '{target}'")
    if target not in df.columns:
        fail(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    drop = [c.strip() for c in drop.split(",") if c.strip()]
    # auto-drop obvious identifier columns
    for c in df.columns:
        if c == target:
            continue
        idlike_name = c.strip().lower() in {"id", "index", "unnamed: 0", "rownumber"}
        idlike_text = (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique() == len(df)
        if (idlike_name or idlike_text) and c not in drop:
            drop.append(c)
            print(f"Auto-dropping identifier-like column: '{c}'")
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.dropna(axis=1, how="all").dropna(subset=[target])

    # drop constant feature columns (no information)
    const = [c for c in df.columns if c != target and df[c].nunique(dropna=True) <= 1]
    if const:
        print(f"Dropping constant columns: {const}")
        df = df.drop(columns=const)

    n_classes = df[target].nunique()
    if n_classes < 2:
        fail("Target has fewer than 2 classes.")
    if n_classes > MAX_CLASSES:
        fail(f"Target '{target}' has {n_classes} distinct values; this looks like regression, "
             f"not classification. Pick a different --target.")

    # classes too small to stratify-split
    counts = df[target].value_counts()
    tiny = counts[counts < 5].index
    if len(tiny):
        print(f"Removing classes with <5 rows: {list(tiny)}")
        df = df[~df[target].isin(tiny)]

    n_feat, n_rows = df.shape[1] - 1, len(df)
    print(f"Usable data -> instances: {n_rows} | features: {n_feat} | classes: {df[target].nunique()}")
    if n_feat < MIN_FEATURES:
        fail(f"Only {n_feat} features (minimum {MIN_FEATURES}).")
    if n_rows < MIN_INSTANCES:
        fail(f"Only {n_rows} instances (minimum {MIN_INSTANCES}).")
    return df, target


def build_models(num_cols, cat_cols, seed):
    def prep(scale):
        num_steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("sc", StandardScaler()))
        return ColumnTransformer([
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             cat_cols),
        ])

    def pipe(scale, clf):
        return Pipeline([("prep", prep(scale)), ("clf", clf)])

    return {
        "Logistic Regression": pipe(True, LogisticRegression(max_iter=2000, random_state=seed)),
        "Decision Tree": pipe(False, DecisionTreeClassifier(max_depth=6, random_state=seed)),
        "KNN": pipe(True, KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": pipe(False, GaussianNB()),
        "Support Vector Machine": pipe(True, SVC(kernel="rbf", probability=True, random_state=seed)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to any CSV file")
    ap.add_argument("--target", default=None, help="target column (default: last column)")
    ap.add_argument("--drop", default="", help="comma-separated columns to ignore")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, target = prepare(pd.read_csv(args.data), args.target, args.drop)
    X, y_raw = df.drop(columns=[target]), df[target]
    le = LabelEncoder().fit(y_raw.astype(str))
    y = pd.Series(le.transform(y_raw.astype(str)), index=df.index)
    classes = list(le.classes_)

    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_tr, X_te, y_tr, y_te, _, raw_te = train_test_split(
        X, y, y_raw, test_size=0.2, stratify=y, random_state=args.seed)

    pd.concat([X_te, raw_te], axis=1).to_csv("test_data.csv", index=False)
    json.dump({"features": list(X.columns), "numeric": num_cols, "categorical": cat_cols,
               "target": target, "classes": classes}, open("model/meta.json", "w"), indent=2)

    rows = []
    for name, p in build_models(num_cols, cat_cols, args.seed).items():
        p.fit(X_tr, y_tr)
        joblib.dump(p, f"model/{name.replace(' ', '_').lower()}.joblib")
        m = compute_metrics(y_te, p.predict(X_te), p.predict_proba(X_te), len(classes))
        rows.append({"ML Model Name": name, **m})

    res = pd.DataFrame(rows).round(4)
    res.to_csv("model/metrics.csv", index=False)
    print("\n" + res.to_markdown(index=False))
    print("\nSaved: model/*.joblib, model/meta.json, model/metrics.csv, test_data.csv")


if __name__ == "__main__":
    main()
