#!/usr/bin/env python3
"""Run upstream render.py (test views) and metrics.py unchanged, with physproxy's runtime
Metric3D cache fix. Prints and writes <model>/results.json (written by metrics.py).

  python physproxy/eval/render_metrics.py MODEL_DIR
"""
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()

model = sys.argv[1]
os.chdir(ROOT)
sys.argv = ["render.py", "-m", model, "--skip_train", "--quiet"]
runpy.run_path(os.path.join(ROOT, "render.py"), run_name="__main__")
sys.argv = ["metrics.py", "-m", model]
runpy.run_path(os.path.join(ROOT, "metrics.py"), run_name="__main__")
