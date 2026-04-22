"""Family-agnostic loader for daily direction models.

``load_any_model(path)`` reads the pickle, inspects its ``family`` field, and
dispatches to the right class. Legacy XGB pickles (without a ``family`` key)
fall back to ``XGBStockModel.load``.

All supported classes expose:
    model.feature_cols : list[str]
    model.predict_scores(df) -> pd.Series  # [0, 1] indexed like df
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def _peek_family(path: Path) -> str:
    """Read just the family tag without deserialising the full pickle twice.

    Falls back to full-deser if the file doesn't have a family key (legacy
    XGB pickles). Callers should treat the returned string as a hint, not a
    guarantee.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return str(data.get("family") or "xgb")


def load_any_model(path: Path | str):
    """Return the right model instance for the pickle at ``path``.

    Dispatch table:
        "xgb" | <no family>   → xgbnew.model.XGBStockModel
        "lgb"                 → xgbnew.model_lgb.LGBMStockModel
        "cat"                 → xgbnew.model_cat.CatBoostStockModel
        "mlp"                 → xgbnew.model_mlp.MLPStockModel
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    family = _peek_family(path)
    if family == "xgb":
        from xgbnew.model import XGBStockModel
        return XGBStockModel.load(path)
    if family == "lgb":
        from xgbnew.model_lgb import LGBMStockModel
        return LGBMStockModel.load(path)
    if family == "cat":
        from xgbnew.model_cat import CatBoostStockModel
        return CatBoostStockModel.load(path)
    if family == "mlp":
        from xgbnew.model_mlp import MLPStockModel
        return MLPStockModel.load(path)
    if family == "mlp_muon":
        from xgbnew.model_mlp_muon import MuonMLPStockModel
        return MuonMLPStockModel.load(path)
    raise ValueError(f"Unknown model family {family!r} in {path}")


def infer_family_from_model(model) -> str:
    """Best-effort: read the declared family, else 'xgb' for legacy."""
    fam = getattr(model, "family", None)
    return str(fam) if fam else "xgb"


__all__ = ["load_any_model", "infer_family_from_model"]
