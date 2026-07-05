#!/usr/bin/env python3
"""
End-to-end orchestrator for the Madrid rental anomaly detector.

    python run_pipeline.py            # analyse existing data + build report
    python run_pipeline.py --scrape   # (re)scrape Fotocasa first, then analyse

Stages:
    1. (optional) scrape Fotocasa Madrid rentals   -> data/listings.csv
    2. feature engineering + spatial join          (in-memory)
    3. isolation forest + comparables + hedonic     -> output/*.csv
    4. HTML report                                  -> output/report.html
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def main():
    if "--scrape" in sys.argv:
        print(">>> Stage 1: scraping Fotocasa (this takes a few minutes)")
        subprocess.run([sys.executable, str(SRC / "scrape_fotocasa.py"), "3", "4.5"],
                       check=True)

    if not (ROOT / "data" / "listings.csv").exists():
        sys.exit("No data/listings.csv found. Run with --scrape first.")

    print(">>> Stage 3: isolation forest + comparables + hedonic regression")
    import analyze
    analyze.run()

    print(">>> Stage 4: building HTML report")
    import report
    report.build()

    print("\nDone. Open output/report.html")


if __name__ == "__main__":
    main()
