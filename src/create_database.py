from __future__ import annotations

from typing import Optional

from sqlalchemy.engine import Engine

from src.models.models import Base as ModelsBase
from src.portfolio_risk import Base as PortfolioBase, _get_engine


def create_all(engine: Optional[Engine] = None) -> None:
    """Create all SQLAlchemy tables used by the trading system."""
    resolved_engine = engine or _get_engine()
    for metadata in (ModelsBase.metadata, PortfolioBase.metadata):
        metadata.create_all(resolved_engine)


if __name__ == "__main__":
    create_all()
