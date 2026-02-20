import os
import joblib
import xgboost as xgb


def train_xgb(
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    params=None,
    model_path=None,
    verbose=100,
):
    if params is None:
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=verbose,
    )

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    return model
