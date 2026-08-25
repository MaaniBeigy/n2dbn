"""
utils/logging.py
================
Structured JSON logging for N²DBN experiments.

Features
--------
* Structured JSON logs per run (one JSON line per event).
* Optional MLflow / W&B integration (graceful no-op if not installed).
* RunLogger context manager that auto-closes the log file on exit.

Usage
-----
    from src.utils.logging import RunLogger

    with RunLogger("logs/run_001.jsonl", tags={"scenario": "sanity_L3"}) as log:
        log.log_scalar("ELBO", -1234.5, step=0)
        log.log_scalars({"SHD_k1": 3, "F1_k1": 0.82}, step=1)
        log.log_config(cfg.__dict__)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# JSON line logger
# ---------------------------------------------------------------------------


class RunLogger:
    """
    Append-mode structured JSON logger.

    Each call to `log_*` writes one JSON line to the log file with a
    timestamp, the event type, and the payload.

    Parameters
    ----------
    path   : file path (directory is created if absent)
    tags   : static key-value pairs added to every log line
    use_mlflow : if True and mlflow is installed, mirror scalars to MLflow
    use_wandb  : if True and wandb is installed, mirror scalars to W&B
    """

    def __init__(
        self,
        path: str,
        tags: Optional[Dict[str, Any]] = None,
        use_mlflow: bool = False,
        use_wandb: bool = False,
    ) -> None:
        self.path = path
        self.tags = tags or {}
        self._use_mlflow = use_mlflow
        self._use_wandb = use_wandb
        self._file = None
        self._mlflow = None
        self._wandb = None
        self._start_time = time.time()

    # ------------------------------------------------------------------
    def __enter__(self) -> "RunLogger":
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")
        if self._use_mlflow:
            try:
                import mlflow

                self._mlflow = mlflow
            except ImportError:
                pass
        if self._use_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                pass
        self._write({"event": "run_start", "tags": self.tags})
        return self

    def __exit__(self, *args) -> None:
        self._write({"event": "run_end", "elapsed_s": time.time() - self._start_time})
        if self._file is not None:
            self._file.close()

    # ------------------------------------------------------------------
    def log_scalar(self, name: str, value: float, step: int = 0) -> None:
        """Log a single scalar value."""
        self._write({"event": "scalar", "name": name, "value": value, "step": step})
        if self._mlflow:
            self._mlflow.log_metric(name, value, step=step)
        if self._wandb:
            self._wandb.log({name: value}, step=step)

    def log_scalars(self, scalars: Dict[str, float], step: int = 0) -> None:
        """Log multiple scalar values at the same step."""
        self._write({"event": "scalars", "values": scalars, "step": step})
        if self._mlflow:
            self._mlflow.log_metrics(scalars, step=step)
        if self._wandb:
            self._wandb.log(dict(scalars, **{"_step": step}))

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log hyperparameter / config dict."""
        self._write({"event": "config", "config": config})
        if self._mlflow:
            self._mlflow.log_params(
                {k: str(v) for k, v in config.items() if not isinstance(v, dict)}
            )
        if self._wandb:
            self._wandb.config.update(config)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Record the path of a saved artifact (model checkpoint, CSV, etc.)."""
        self._write(
            {"event": "artifact", "path": path, "name": name or os.path.basename(path)}
        )

    def log_text(self, message: str) -> None:
        """Log a free-form text message (e.g. error traceback)."""
        self._write({"event": "text", "message": message})

    # ------------------------------------------------------------------
    def _write(self, payload: Dict[str, Any]) -> None:
        record = {"ts": time.time(), **self.tags, **payload}
        line = json.dumps(record, default=str) + "\n"
        if self._file is not None:
            self._file.write(line)
            self._file.flush()


# ---------------------------------------------------------------------------
# Convenience: read a JSONL log file back as list of dicts
# ---------------------------------------------------------------------------


def load_log(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL log file as a list of event dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_scalars(log: List[Dict[str, Any]], name: str) -> List[Dict]:
    """Extract all scalar events with the given name from a log."""
    return [r for r in log if r.get("event") == "scalar" and r.get("name") == name]
