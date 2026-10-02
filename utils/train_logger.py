#
# Plain-text training logs, so a run on the cluster can be inspected afterwards
# without access to its stdout. Everything is written under <model_path>/logs/.
#

import os
import sys
import socket
import subprocess
from datetime import datetime

import torch


class _Tee:
    """Duplicates writes to a stream into a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _git(*cmd):
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", repo, *cmd], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class TrainLogger:
    """
    Files written to <model_path>/logs/:
      stdout.txt   - every print and traceback of the run (stdout + stderr)
      config.txt   - command line, git state, machine and all arguments
      metrics.txt  - tab-separated training stats every `metrics_interval` iterations
      eval.txt     - tab-separated test/train evaluation results
      summary.txt  - final model size, timings and output paths
    """

    def __init__(self, model_path, metrics_interval=100):
        self.log_dir = os.path.join(model_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_interval = metrics_interval
        self.start_time = datetime.now()

        # Capture all prints and uncaught exceptions. Line buffering so the file
        # is up to date if the job gets killed.
        self._stdout_file = open(os.path.join(self.log_dir, "stdout.txt"), "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, self._stdout_file)
        sys.stderr = _Tee(sys.__stderr__, self._stdout_file)

        self._metrics_file = open(os.path.join(self.log_dir, "metrics.txt"), "a", buffering=1)
        self._metrics_file.write("iteration\ttime_s\tloss\tema_loss\tpixel_loss\tnormal_loss\tweight_loss"
                                 "\tsigma\tnum_vertices\tnum_triangles\tgpu_mem_gb\n")

        self._eval_file = open(os.path.join(self.log_dir, "eval.txt"), "a", buffering=1)
        self._eval_file.write("iteration\tsplit\tl1\tpsnr\tssim\tlpips\tfps\n")

    def elapsed(self):
        return (datetime.now() - self.start_time).total_seconds()

    def write_config(self, args):
        with open(os.path.join(self.log_dir, "config.txt"), "w") as f:
            f.write("start_time: {}\n".format(_now()))
            f.write("host: {}\n".format(socket.gethostname()))
            f.write("command: {}\n".format(" ".join(sys.argv)))
            f.write("git_branch: {}\n".format(_git("rev-parse", "--abbrev-ref", "HEAD")))
            f.write("git_commit: {}\n".format(_git("rev-parse", "HEAD")))
            dirty = _git("status", "--porcelain", "--untracked-files=no")
            f.write("git_dirty: {}\n".format("unknown" if dirty == "unknown" else ("yes" if dirty else "no")))
            if torch.cuda.is_available():
                f.write("gpu: {}\n".format(torch.cuda.get_device_name(0)))
            f.write("torch: {}\n".format(torch.__version__))
            f.write("\n[arguments]\n")
            for key, value in sorted(vars(args).items()):
                f.write("{}: {}\n".format(key, value))

    def log_metrics(self, iteration, loss, ema_loss, pixel_loss, normal_loss, weight_loss, sigma, triangles):
        if iteration % self.metrics_interval != 0:
            return
        gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        self._metrics_file.write("{}\t{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\t{}\t{:.2f}\n".format(
            iteration, self.elapsed(), float(loss), float(ema_loss), float(pixel_loss), float(normal_loss),
            float(weight_loss), float(sigma), triangles.vertices.shape[0], triangles._triangle_indices.shape[0], gpu_mem))

    def log_eval(self, iteration, split, l1, psnr, ssim, lpips, fps):
        self._eval_file.write("{}\t{}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.1f}\n".format(
            iteration, split, float(l1), float(psnr), float(ssim), float(lpips), float(fps)))

    def write_summary(self, items):
        with open(os.path.join(self.log_dir, "summary.txt"), "w") as f:
            f.write("end_time: {}\n".format(_now()))
            f.write("total_time_min: {:.1f}\n".format(self.elapsed() / 60.0))
            for key, value in items.items():
                f.write("{}: {}\n".format(key, value))
