import sys, re, pathlib

metrics_file = pathlib.Path("metrics.txt")
if not metrics_file.exists():
    print("metrics.txt not found. Did training run?")
    sys.exit(2)

txt = metrics_file.read_text()
m = re.search(r"R2:\s*([0-9.]+)", txt, re.I)
if not m:
    print("Could not parse R2 from metrics.txt")
    sys.exit(2)

current = float(m.group(1))
thres = float(pathlib.Path("expected_min_r2.txt").read_text().strip())

print(f"Current R2={current:.4f} / Threshold={thres:.4f}")
if current < thres:
    print("❌ Gate failed: R2 below threshold.")
    sys.exit(1)

print("✅ Gate passed.")

