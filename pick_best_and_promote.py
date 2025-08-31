#!/usr/bin/env python3
"""
Pick best run by r2_score and promote its registered model version.

- No server-side filter syntax (filters/sorts in Python).
- Uses aliases (preferred). Adds a tag instead of PATCHing the description.
- Optional fallback to legacy stages: set ALLOW_STAGE_FALLBACK=1.
"""

import os, sys
from urllib.parse import urlparse
import requests

# ==== Config (env-overridable) ====
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://54.91.101.120:8081")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "week6-Sunday-Exp2")
MODEL_NAME = os.getenv("MODEL_NAME", "w6-simple-regressor")
PROMOTE_ALIAS = os.getenv("PROMOTE_ALIAS", "staging")
ALLOW_STAGE_FALLBACK = os.getenv("ALLOW_STAGE_FALLBACK", "0") == "1"
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SEC", "15"))
DISABLE_PROXIES = os.getenv("DISABLE_PROXIES", "1") == "1"  # default ON
# ==================================

# Fail fast for MLflow internal HTTP
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(HTTP_TIMEOUT)))

def _disable_env_proxies():
    if DISABLE_PROXIES:
        for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"):
            os.environ.pop(k, None)

def _ensure_no_proxy_for(uri: str):
    p = urlparse(uri)
    host = p.hostname or ""
    port = p.port or (443 if p.scheme == "https" else 80)
    additions = [host, f"{host}:{port}", "127.0.0.1", "localhost"]
    for var in ("NO_PROXY","no_proxy"):
        cur = [x for x in (os.environ.get(var,"").split(",")) if x]
        merged = []
        for x in cur + additions:
            if x and x not in merged:
                merged.append(x)
        os.environ[var] = ",".join(merged)

def _reachable(uri: str) -> bool:
    base = uri.rstrip("/")
    try:
        r = requests.get(base + "/api/2.0/mlflow/experiments/get-by-name",
                         params={"experiment_name": EXPERIMENT_NAME},
                         timeout=HTTP_TIMEOUT)
        return r.status_code < 500  # any non-5xx means “server reachable”
    except Exception:
        return False

# Apply proxy hygiene BEFORE importing mlflow
_disable_env_proxies()
_ensure_no_proxy_for(TRACKING_URI)

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def main():
    print(f"[INFO] TRACKING_URI = {TRACKING_URI}")
    print(f"[INFO] NO_PROXY     = {os.environ.get('NO_PROXY','')}")
    if not _reachable(TRACKING_URI):
        raise SystemExit(
            f"[ERR] MLflow URI not reachable from this process: {TRACKING_URI}\n"
            f"Try:\n  curl -i {TRACKING_URI.rstrip('/')}/api/2.0/mlflow/experiments/get-by-name"
            f" --get --data-urlencode 'experiment_name={EXPERIMENT_NAME}'"
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # 1) Find experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"[ERR] Experiment '{EXPERIMENT_NAME}' not found at {TRACKING_URI}")

    # 2) Pull runs without server-side filter; filter/sort locally
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        # no filter_string here (avoid IS [NOT] NULL incompatibility)
        # also skip order_by; we’ll sort locally for maximum compatibility
        max_results=1000,
    )
    if runs.empty or "metrics.r2_score" not in runs.columns:
        raise SystemExit("[ERR] No runs with metric 'r2_score' found.")

    runs = runs.dropna(subset=["metrics.r2_score"])
    if runs.empty:
        raise SystemExit("[ERR] All runs have null 'r2_score'.")

    runs = runs.sort_values(by="metrics.r2_score", ascending=False, kind="mergesort")
    best = runs.iloc[0]
    best_run_id = best.run_id
    best_r2 = float(best["metrics.r2_score"])
    print(f"[INFO] Best run: {best_run_id}  r2={best_r2:.4f}")

    # 3) Map run_id -> registered model version
    versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    best_version = next((int(v.version) for v in versions if v.run_id == best_run_id), None)
    if best_version is None:
        raise SystemExit(f"[ERR] No registered version under '{MODEL_NAME}' for run_id {best_run_id}.")

    # 4) Promote via alias + annotate with a tag (avoids PATCH endpoints)
    try:
        client.set_registered_model_alias(MODEL_NAME, PROMOTE_ALIAS, best_version)
        client.set_model_version_tag(
            MODEL_NAME, str(best_version), "promotion_note",
            f"Auto-promoted via alias '{PROMOTE_ALIAS}' (r2_score={best_r2:.4f})"
        )
        try:
            aliased = client.get_model_version_by_alias(MODEL_NAME, PROMOTE_ALIAS)
            print(f"[OK] Alias '{PROMOTE_ALIAS}' -> {MODEL_NAME} v{aliased.version} (r2={best_r2:.4f}).")
        except Exception:
            print(f"[OK] Alias '{PROMOTE_ALIAS}' -> {MODEL_NAME} v{best_version} (r2={best_r2:.4f}).")

    except MlflowException as e:
        if ALLOW_STAGE_FALLBACK and ("alias" in str(e).lower() or "404" in str(e)):
            print("[WARN] Aliases unsupported; falling back to stage 'Staging'…")
            client.transition_model_version_stage(
                name=MODEL_NAME, version=str(best_version),
                stage="Staging", archive_existing_versions=True
            )
            client.set_model_version_tag(
                MODEL_NAME, str(best_version), "promotion_note",
                f"Promoted to Staging (r2_score={best_r2:.4f})"
            )
            print(f"[OK] Promoted {MODEL_NAME} v{best_version} to Staging (r2={best_r2:.4f}).")
        else:
            raise

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[FATAL] {ex}", file=sys.stderr)
        sys.exit(2)

