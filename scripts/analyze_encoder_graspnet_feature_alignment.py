#!/usr/bin/env python3
"""兼容入口：转发至 `feature_alignment_analysis.py`。"""
import os
import runpy

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(os.path.join(_here, "feature_alignment_analysis.py"), run_name="__main__")
