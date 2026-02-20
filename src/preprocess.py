import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, test_size=0.3, val_size=0.5, random_state=42, scaler_path=None):
    df_processed = pd.get_dummies(df, columns=["Season"], prefix="Season")
    df_processed["is_weekend"] = df_processed["Day of Week"].apply(lambda x: 1 if x >= 5 else 0)

    X = df_processed.drop(["Timestamp", "Load Demand (kW)"], axis=1)
    y = df_processed["Load Demand (kW)"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return {
        "df_processed": df_processed,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }
