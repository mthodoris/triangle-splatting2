"""Runtime-only workarounds for physproxy's own entry points (upstream files untouched).

use_cached_metric3d(): upstream scene/dataset_readers.py calls
torch.hub.load('yvanyin/metric3d', ...) on every scene load. Without an explicit
ref, torch.hub asks GitHub for the default branch even when the model is
cached, and a GitHub outage (HTTP 504, seen 2026-11-10) aborts training. In this
process only, redirect that call to the cached 'main' checkout without contacting
GitHub. If no cache exists, the original call is made unchanged.
"""
import os

import torch


def use_cached_metric3d():
    if getattr(torch.hub, "_physproxy_patched", False):
        return
    orig = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        cache = os.path.join(torch.hub.get_dir(), "yvanyin_metric3d_main")
        if repo_or_dir == "yvanyin/metric3d" and os.path.isdir(cache) and kwargs.get("source", "github") == "github":
            kwargs.setdefault("skip_validation", True)
            return orig("yvanyin/metric3d:main", model, *args, **kwargs)
        return orig(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    torch.hub._physproxy_patched = True
