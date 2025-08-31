import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# --- Settings ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "w6-d4-registry-demo"
REGISTERED_NAME = "w6-simple-regressor"   # One registry name -> multiple versions


mlflow.set_tracking_uri("http://54.91.101.120:8081/")
mlflow.set_experiment("week6-Sunday-Exp2")


# Data
data = pd.read_csv("data.csv")
X = data[["feature"]]
y = data["target"]

# Candidate models
experiments = [
    {"model": LinearRegression(), "name": "LinearRegression"},
    {"model": Ridge(alpha=0.5), "name": "Ridge"},
    {"model": Lasso(alpha=0.1), "name": "Lasso"}
]

# Autolog + explicit logging
mlflow.sklearn.autolog()

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]) as run:
        model = exp["model"].fit(X, y)
        r2 = model.score(X, y)

        # manual logs (nice for clarity)
        mlflow.log_param("model_type", exp["name"])
        mlflow.log_metric("r2_score", r2)

        # save locally (not required, but handy)
        joblib.dump(model, f"{exp['name']}.pkl")

        # create a small signature so consumers know expected input schema
        signature = infer_signature(X, model.predict(X))

        # This line logs AND REGISTERS a new Model Version under REGISTERED_NAME
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME,
            signature=signature,
            input_example=X.head(2)
        )

        print(f"Logged {exp['name']} | run_id={run.info.run_id} | r2={r2:.4f}")

