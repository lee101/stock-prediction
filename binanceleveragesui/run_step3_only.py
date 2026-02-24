#!/usr/bin/env python3
"""Run just step 3 (policy sweep) of the retrain pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binanceleveragesui.retrain_full_pipeline import step3_retrain_policies
step3_retrain_policies()
