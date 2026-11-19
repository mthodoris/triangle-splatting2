"""JSON-lines logger for physproxy terms (TensorBoard is not installed here).

One record per call, appended to <model_path>/physproxy_log.jsonl:
    {"iter": 1200, "tag": "loss", "L_surf": 0.012, ...}
Tensors are converted with .item(); the file is flushed on each write so a
crashed run keeps its log. Read back with `read_log(path)`.
"""
import json
import os
import time


def _to_py(v):
    if hasattr(v, "item") and getattr(v, "numel", lambda: 1)() == 1:
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


class JsonLogger:
    def __init__(self, model_path, filename="physproxy_log.jsonl", config=None):
        os.makedirs(model_path, exist_ok=True)
        self.path = os.path.join(model_path, filename)
        self._fh = open(self.path, "a", buffering=1)
        self._t0 = time.time()
        if config is not None:
            self.log(0, "config", **{k: _to_py(v) for k, v in dict(config).items()})

    def log(self, iteration, tag, **values):
        rec = {"iter": int(iteration), "tag": tag, "t": round(time.time() - self._t0, 3)}
        rec.update({k: _to_py(v) for k, v in values.items()})
        self._fh.write(json.dumps(rec) + "\n")

    def close(self):
        self._fh.close()


def read_log(path, tag=None):
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    return [r for r in recs if tag is None or r["tag"] == tag]
