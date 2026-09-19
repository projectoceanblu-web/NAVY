"""SQLAlchemy engine/session plumbing.

Sync engine on purpose: the SAR pipeline is CPU-bound NumPy work and the API
endpoints are declared ``def`` so Starlette runs them in a threadpool.  The AIS
ingester is async but hands its batched writes to ``asyncio.to_thread``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from bob_sentinel.config import get_settings

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _is_serverless() -> bool:
    """Vercel (and most FaaS) set this; a container may serve one request."""
    return bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().sqlalchemy_url
        kwargs: dict = {"pool_pre_ping": True, "future": True}
        connect_args: dict = {}

        if _is_serverless():
            # A serverless container is frozen between invocations, so a
            # pooled connection is usually dead by the next request and just
            # consumes one of the database's limited slots. Open per request.
            kwargs["poolclass"] = NullPool
        else:
            kwargs.update(pool_size=5, max_overflow=10)

        if ":6543" in url:
            # Supabase's transaction-mode pooler multiplexes one server
            # connection across clients, so server-side prepared statements
            # (psycopg3's default above a threshold) collide across sessions.
            connect_args["prepare_threshold"] = None

        if connect_args:
            kwargs["connect_args"] = connect_args
        _engine = create_engine(url, **kwargs)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(), expire_on_commit=False, future=True
        )
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on failure."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session
