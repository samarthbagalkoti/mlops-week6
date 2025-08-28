import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import mlflow

# Point MLflow client to your server
mlflow.set_tracking_uri("http://54.91.173.39:8081/")
mlflow.set_experiment("week6")

# Load dataset
with mlflow.start_run():
    # Load dataset
    data = pd.read_csv("data.csv")
    X = data[["feature"]]
    y = data["target"]

    # Train simple model
    model = LinearRegression()
    model.fit(X, y)

    # Save model locally
    joblib.dump(model, "model.pkl")

    # Custom metric logging (in addition to autolog)
    r2_score = model.score(X, y)
    mlflow.log_metric("r2_score", r2_score)

    # Log params
    mlflow.log_param("model_type", "LinearRegression")

    # Log artifact (metrics file)
    with open("metrics.txt", "w") as f:
        f.write(f"R2: {r2_score}\n")
    mlflow.log_artifact("metrics.txt")
