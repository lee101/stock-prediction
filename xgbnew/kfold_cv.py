#!/usr/bin/env python3
"""Compatibility entrypoint for XGB k-fold validation.

The maintained implementation lives in ``xgbnew.eval_kfold``. This wrapper
keeps existing runbooks that call ``python xgbnew/kfold_cv.py`` working.
"""

from __future__ import annotations

from xgbnew.eval_kfold import main


if __name__ == "__main__":
    raise SystemExit(main())
