# pipeline/exporter.py
"""
Persist pipeline results to disk.
  - results.json  – full structured output (easy to reload)
  - results.csv   – flat table for spreadsheets / pandas
"""

import csv
import json
import pathlib
from datetime import datetime


def save(results: list[dict], out_dir: str = "outputs") -> dict[str, str]:
    """
    Save results list to JSON + CSV.

    Parameters
    ----------
    results : list[dict]   one dict per processed sample
    out_dir : str          directory to write into (created if absent)

    Returns
    -------
    {"json": path_str, "csv": path_str}
    """
    base = pathlib.Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = base / f"results_{ts}.json"
    csv_path  = base / f"results_{ts}.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV – flatten nested dicts (location, etc.)
    flat = [_flatten(r) for r in results]
    if flat:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)

    print(f"📄 Results saved → {json_path}")
    print(f"📄 Results saved → {csv_path}")
    return {"json": str(json_path), "csv": str(csv_path)}


def _flatten(record: dict) -> dict:
    """Collapse nested dicts one level deep for CSV."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                out[f"{k}_{sub_k}"] = sub_v
        else:
            out[k] = v
    return out