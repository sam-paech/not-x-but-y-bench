# src/io_utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

_LOCK = threading.Lock()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def ensure_model_header(results_path: Path, model_id: str, endpoint: str, params: Dict[str, Any], started_at: str) -> None:
    with _LOCK:
        data = _read_json(results_path)
        if model_id not in data:
            data[model_id] = {
                "test_model": model_id,
                "endpoint": endpoint,
                "params": params,
                "started_at": started_at,
                "completed_at": None,
                "samples": [],
                "summary": {"total_prompts": 0, "total_chars": 0, "total_hits": 0, "rate_per_1k": 0.0},
            }
            _atomic_write(results_path, data)


def _recompute_summary(model_blob: Dict[str, Any]) -> None:
    samples = model_blob.get("samples", [])
    chars = sum(s.get("chars", 0) for s in samples if not s.get("error"))
    hits = sum(s.get("hits", 0) for s in samples if not s.get("error"))
    rate = (hits * 1000.0 / chars) if chars > 0 else 0.0
    model_blob["summary"] = {
        "total_prompts": len(samples),
        "total_chars": chars,
        "total_hits": hits,
        "rate_per_1k": rate,
    }


def atomic_update_model_results(results_path: Path, model_id: str, new_sample: Optional[Dict[str, Any]], completed_at: Optional[str] = None) -> None:
    with _LOCK:
        data = _read_json(results_path)
        model_blob = data.setdefault(model_id, {
            "test_model": model_id,
            "endpoint": None,
            "params": {},
            "started_at": None,
            "completed_at": None,
            "samples": [],
            "summary": {"total_prompts": 0, "total_chars": 0, "total_hits": 0, "rate_per_1k": 0.0},
        })

        if new_sample is not None:
            # replace if same prompt_index exists
            if "prompt_index" in new_sample:
                pi = new_sample["prompt_index"]
                replaced = False
                for i, s in enumerate(model_blob["samples"]):
                    if s.get("prompt_index") == pi:
                        model_blob["samples"][i] = new_sample
                        replaced = True
                        break
                if not replaced:
                    model_blob["samples"].append(new_sample)
            else:
                model_blob["samples"].append(new_sample)

        if completed_at is not None:
            model_blob["completed_at"] = completed_at

        _recompute_summary(model_blob)
        _atomic_write(results_path, data)
