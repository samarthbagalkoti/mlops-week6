#!/usr/bin/env python3
# Load the current "staging" model by NAME (no local paths), then predict.

import os
from urllib.parse import urlparse
import pandas as pd
import mlflow

# --- Config (env-overridable) ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://54.91.101.120:8081")
MODEL_NAME   = os.getenv("MODEL_NAME", "w6-simple-regressor")
ALIAS        = os.getenv("MODEL_ALIAS", "staging")  # modern approach
STAGE        = os.getenv("MODEL_STAGE", "Staging")  # legacy fallback
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SEC", "15"))
DISABLE_PROXIES = os.getenv("DISABLE_PROXIES", "1") == "1"
# ---------------------------------

# Make MLflow HTTP fail fast
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(HTTP_TIMEOUT))

# Proxy hygiene (helps when corp proxies are set)
if DISABLE_PROXIES:
    for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"):
        os.environ.pop(k, None)

# Ensure NO_PROXY for the MLflow host
p = urlparse(TRACKING_URI)
host = p.hostname or "localhost"
port = p.port or (443 if p.scheme == "https" else 80)
cur = [x for x in (os.environ.get("NO_PROXY","").split(",")) if x]
for extra in (host, f"{host}:{port}", "127.0.0.1", "localhost"):
    if extra not in cur:
        cur.append(extra)
os.environ["NO_PROXY"] = ",".join(cur)

mlflow.set_tracking_uri(TRACKING_URI)

def load_model_by_alias_then_stage():
    # Preferred: alias (modern)
    try:
        uri = f"models:/{MODEL_NAME}@{ALIAS}"
        model = mlflow.pyfunc.load_model(uri)
        print(f"[OK] Loaded '{MODEL_NAME}' via alias '@{ALIAS}'")
        return model
    except Exception as e_alias:
        print(f"[WARN] Alias '@{ALIAS}' load failed: {e_alias}\nTrying stage '{STAGE}'...")
        # Fallback: legacy stage
        try:
            uri = f"models:/{MODEL_NAME}/{STAGE}"
            model = mlflow.pyfunc.load_model(uri)
            print(f"[OK] Loaded '{MODEL_NAME}' via stage '{STAGE}'")
            return model
        except Exception as e_stage:
            raise SystemExit(
                f"[FATAL] Could not load model '{MODEL_NAME}' via alias '@{ALIAS}' "
                f"or stage '{STAGE}'.\nAlias error: {e_alias}\nStage error: {e_stage}"
            )

if __name__ == "__main__":
    model = load_model_by_alias_then_stage()

    # Small test
    df = pd.DataFrame({"feature": [5, 10]})
    preds = model.predict(df)

    print("Input:\n", df)
    print("Predictions:\n", preds)

