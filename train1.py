import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import yaml  # <-- new

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)
fit_intercept = bool(params["train"]["fit_intercept"])

# Load dataset
data = pd.read_csv("data.csv")
X = data[["feature"]]
y = data["target"]

# Train simple model (now uses param)
model = LinearRegression(fit_intercept=fit_intercept)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"R2: {model.score(X, y)}\n")

