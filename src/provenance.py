"""Run provenance and determinism.

Every result file must be traceable to exactly the model, weights, prompt
template, code, and seed that produced it. A number without this block is not a
result (DECISIONS.md D-011).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED_DEFAULT = 0


def set_determinism(seed: int = SEED_DEFAULT) -> dict[str, Any]:
    """Pin every RNG we can reach. Returns what was actually applied."""
    applied: dict[str, Any] = {"seed": seed}

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np

        np.random.seed(seed)
        applied["numpy"] = np.__version__
    except ImportError:  # pragma: no cover - numpy is a hard dep
        applied["numpy"] = None

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Single-threaded matmul removes one more source of run-to-run drift on
        # CPU; the models here are small enough that the cost is acceptable.
        torch.set_num_threads(1)
        applied["torch"] = torch.__version__
        applied["torch_deterministic_algorithms"] = True
        applied["torch_num_threads"] = 1
    except ImportError:
        applied["torch"] = None

    return applied


def git_sha(short: bool = False) -> str | None:
    """The code commit this run came from, or None outside a repo."""
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        out = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_items(item_ids: Iterable[str]) -> str:
    """Stable 12-hex digest of WHICH items were scored."""
    blob = "\n".join(sorted(item_ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def hash_prompts(prompts: Iterable[str]) -> str:
    """Stable 12-hex digest of the exact TEXT the model saw.

    `hash_items` records only which ids were in the run, so it cannot tell you
    whether a passage was edited afterwards. This can. It is the difference
    between "the same 80 items" and "the same 80 prompts", and only the second
    licenses quoting a stored run against the current working tree.
    `scripts/verify_run.py` recomputes it.
    """
    blob = "\x00".join(sorted(prompts)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_revision(model_name: str, model=None) -> str | None:
    """The resolved HF commit hash for the weights actually loaded.

    Tries the loaded config first (offline-safe), then the Hub API.
    """
    for obj in (model, getattr(model, "config", None)):
        sha = getattr(obj, "_commit_hash", None)
        if isinstance(sha, str) and sha:
            return sha
    try:
        from huggingface_hub import model_info

        return model_info(model_name).sha
    except Exception:  # noqa: BLE001 - offline or private repo; not fatal
        return None


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ("torch", "transformers", "numpy", "scipy", "sklearn", "pydantic"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None
    return env


def run_meta(**extra: Any) -> dict[str, Any]:
    """The provenance block written at the top of every result file."""
    meta: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "code_git_sha": git_sha(),
        "code_git_dirty": git_dirty(),
        "environment": environment(),
    }
    meta.update(extra)
    return meta


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return p
