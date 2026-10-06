#!/usr/bin/env python3
"""Thin entry point: ``python3 tre/eval/report.py <arm_dir>... --out DIR`` (see tre_eval/report.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tre_eval.report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
