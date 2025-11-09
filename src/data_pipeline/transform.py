import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def transform_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "Disease"]

    # Symptom → 0/1
    for col in feature_cols:
        df[col] = df[col].notnull().astype(int)

    X = df[feature_cols]
    y_raw = df["Disease"]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name="target", index=df.index)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert y to DataFrame for uniform saving
    splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_frame(),
        "y_test": y_test.to_frame()
    }

    return splits, mapping
