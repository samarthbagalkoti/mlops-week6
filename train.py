import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import mlflow

# Point MLflow client to your server
mlflow.set_tracking_uri("http://54.91.173.39:8081/")
mlflow.set_experiment("week6")

# Load dataset
data = pd.read_csv("data.csv")
X = data[["feature"]]
y = data["target"]

# Train simple model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

# Log with MLflow
with mlflow.start_run():
    # metrics
    r2 = model.score(X, y)
    mlflow.log_metric("R2", r2)

    # log artifact
    mlflow.log_artifact("model.pkl")

    # optionally log params
    mlflow.log_param("fit_intercept", model.fit_intercept)

