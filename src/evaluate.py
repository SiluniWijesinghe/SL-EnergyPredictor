import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test_scaled, y_test):
    preds = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return {"preds": preds, "rmse": rmse, "r2": r2}


def build_importance_df(model, feature_names, importance_type="gain"):
    booster = model.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    importance_map = {}
    for i, name in enumerate(feature_names):
        importance_map[name] = scores.get(f"f{i}", 0.0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": [importance_map[name] for name in feature_names],
        }
    )
    return df.sort_values("importance", ascending=True)
