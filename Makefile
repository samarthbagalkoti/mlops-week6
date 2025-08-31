.PHONY: setup test repro train register promote predict gate ci clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

# If you wired W6:D1 dvc.yaml with 'train.py' as a stage, this reproduces pipeline.
repro:
	dvc repro || true

# Explicit training+registration (W6:D4)
register:
	python train_register.py

promote:
	python pick_best_and_promote.py

predict:
	python load_staging_and_predict.py

# Gate on R2 written by your training script into metrics.txt
gate:
	python gate_r2.py

ci: setup test repro register gate

clean:
	rm -rf mlruns mlflow.db .pytest_cache __pycache__ nohup.out

start-mlflow:
	MLFLOW_TRACKING_URI= http://54.91.101.120:8081 nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://$(MLF_ART_BUCKET)/artifacts --host 0.0.0.0 --port 8081 &

dvc-push:
	dvc push

dvc-pull:
	dvc pull
