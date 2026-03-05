from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path


_BASE = Path(__file__).resolve().parents[1] / "historian_agent"
_PKG = types.ModuleType("historian_agent")
_PKG.__path__ = [str(_BASE)]
sys.modules.setdefault("historian_agent", _PKG)


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _BASE / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


orientation_mod = _load("historian_agent.demographic_orientation", "demographic_orientation.py")


def _create_demo_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE people (
                canonical_name TEXT,
                gender TEXT,
                race TEXT,
                ethnicity TEXT,
                national_origin TEXT,
                occupation_primary TEXT,
                occupation_normalized TEXT,
                division TEXT,
                hire_year INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO people (
                canonical_name, gender, race, ethnicity, national_origin,
                occupation_primary, occupation_normalized, division, hire_year
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("A", "male", "white", "irish", "ireland", "brakeman", "brakeman", "east", 1892),
                ("B", "male", "white", "german", "germany", "laborer", "laborer", "west", 1901),
                ("C", "female", "black", "african_american", "usa", "clerk", "clerk", "east", 1908),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_build_demographic_orientation_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "demographics.db"
    _create_demo_db(db_path)

    summary, meta = orientation_mod.build_demographic_orientation(str(db_path), top_n=3)

    assert meta["enabled"] is True
    assert meta["people_total"] == 3
    assert "People records: 3" in summary
    assert "Top occupations:" in summary
    assert "Race distribution (top):" in summary


def test_build_demographic_orientation_missing_db(tmp_path: Path) -> None:
    summary, meta = orientation_mod.build_demographic_orientation(str(tmp_path / "missing.db"), top_n=3)
    assert summary == ""
    assert meta["enabled"] is False
    assert meta["reason"] == "db_missing"
