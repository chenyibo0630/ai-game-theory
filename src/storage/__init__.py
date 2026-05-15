"""Pluggable storage sinks for simulation runs.

Two implementations:

- :class:`JsonFileSink` — writes one JSON file per run to `runs/`. Default for
  local development; matches what `src/main.py` used to do directly.
- :class:`MySQLSink` — appends rows to the MySQL schema in `db/schema.sql`.
  Used by the docker-compose stack so the frontend can query live runs.

Both implement the :class:`RunSink` protocol so `World` doesn't care which one
is active.
"""

from .base import NullSink, RunAgent, RunSink, RunStart
from .json_sink import JsonFileSink

__all__ = ["JsonFileSink", "NullSink", "RunAgent", "RunSink", "RunStart"]

try:  # MySQL is optional — only importable when PyMySQL is installed.
    from .mysql_sink import MySQLSink  # noqa: F401

    __all__.append("MySQLSink")
except ImportError:  # pragma: no cover — exercised only on minimal installs
    pass
