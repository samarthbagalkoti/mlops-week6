import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://54.91.101.120:8081/")
mlflow.set_experiment("week6-Sunday")

# Dataset
data = pd.read_csv("data.csv")
X = data[["feature"]]
y = data["target"]

# Candidate models and params
experiments = [
    {"model": LinearRegression(), "name": "LinearRegression"},
    {"model": Ridge(alpha=0.5), "name": "Ridge"},
    {"model": Lasso(alpha=0.1), "name": "Lasso"}
]

# Autologging enables automatic tracking
mlflow.sklearn.autolog()

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        model = exp["model"]
        model.fit(X, y)

        # Save model locally
        joblib.dump(model, f"{exp['name']}.pkl")

        # Log params
        mlflow.log_param("model_type", exp["name"])

        # Custom metric logging
        r2_score = model.score(X, y)
        mlflow.log_metric("r2_score", r2_score)

        # Save + log metrics file
        with open("metrics.txt", "w") as f:
            f.write(f"Model: {exp['name']}, R2: {r2_score}\n")
        mlflow.log_artifact("metrics.txt")

