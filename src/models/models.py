from typing import Any, Type

from sqlalchemy import Column, String, Float, Sequence, DateTime, func, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base: Type[Any] = declarative_base()


class Trade(Base):
    __tablename__ = 'trades'
    id = Column(BigInteger, Sequence('trade_id_seq'), primary_key=True)
    date = Column(DateTime(timezone=True))
    uid = Column(BigInteger())
    amount = Column(Float())
    price = Column(Float())
    total = Column(Float())

    trade_type = Column(String())
    currency_pair = Column(String())

    create_date = Column(DateTime(timezone=True), default=func.now()),
