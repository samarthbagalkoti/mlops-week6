Start server → mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

Export URI → export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Train & register → python train_register.py

Promote best → python pick_best_and_promote.py

Load staged → python load_staging_and_predict.py


Also refer your Evernote Notes
