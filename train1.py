import json
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---- read params ----
with open("params.yaml") as f:
    p = yaml.safe_load(f)["train"]
fit_intercept = bool(p.get("fit_intercept", True))
test_size     = float(p.get("test_size", 0.2))
random_state  = int(p.get("random_state", 42))

# ---- load data ----
data = pd.read_csv("data.csv")
X = data[["feature"]]
y = data["target"]

# ---- split train/test so metrics can change across runs ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# ---- train ----
model = LinearRegression(fit_intercept=fit_intercept)
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

# ---- evaluate (test set) ----
y_pred = model.predict(X_test)
metrics = {
    "R2":  float(r2_score(y_test, y_pred)),
    "MAE": float(mean_absolute_error(y_test, y_pred)),
}

# ---- write metrics.json ----
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

