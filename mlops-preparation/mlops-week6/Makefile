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

