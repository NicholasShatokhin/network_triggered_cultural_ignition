from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "results" / "reproduced"
OUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["CULTURAL_IGNITION_OUT"] = str(OUT)

derived_archaeology = OUT / "archaeological_single_chain_data.csv"
print("[RUN] prepare_archaeological_data.py", flush=True)
subprocess.run(
    [
        sys.executable,
        str(CODE / "prepare_archaeological_data.py"),
        "--output",
        str(derived_archaeology),
    ],
    check=True,
    env=env,
)
env["CULTURAL_IGNITION_ARCHAEOLOGY_DATA"] = str(derived_archaeology)

scripts = [
    "archaeology_robustness.py",
    "aggregate_ablation.py",
    "analyze_topologies_independent.py",
    "analyse_sensitivity_frozen.py",
    "summarise_scaling_frozen.py",
    "cosmological_embedding.py",
]
for script in scripts:
    print(f"[RUN] {script}", flush=True)
    subprocess.run([sys.executable, str(CODE / script)], check=True, env=env)
print(f"Reproduced published analyses in {OUT}")
