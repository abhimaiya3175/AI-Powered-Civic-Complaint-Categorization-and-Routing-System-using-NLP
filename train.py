#!/usr/bin/env python
# main.py  – single entry point for the Kannada Civic Complaint Pipeline
"""
Usage
-----
  python main.py              # process 15 samples (default in config.py)
  python main.py --samples 5  # process only 5 samples
  python main.py --no-map     # skip Leaflet map generation
"""

import argparse

from pipeline        import run
from pipeline.exporter   import save
from pipeline.map_export import generate_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kannada Civic Complaint Pipeline")
    p.add_argument("--samples", type=int, default=None,
                   help="Number of audio samples to process (default: from config)")
    p.add_argument("--out-dir", default="outputs",
                   help="Directory for JSON, CSV and map outputs")
    p.add_argument("--no-map", action="store_true",
                   help="Skip Leaflet HTML map generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("🚀 Starting Kannada Civic Complaint Pipeline\n")

    # ── 1. Run pipeline ──────────────────────────────────────────────────
    results = run(n_samples=args.samples)

    # ── 2. Save JSON + CSV ───────────────────────────────────────────────
    paths = save(results, out_dir=args.out_dir)

    # ── 3. Generate Leaflet map ──────────────────────────────────────────
    if not args.no_map:
        generate_map(results, out_dir=args.out_dir)

    print("\n✅ All done!")
    print(f"   JSON → {paths['json']}")
    print(f"   CSV  → {paths['csv']}")


if __name__ == "__main__":
    main()