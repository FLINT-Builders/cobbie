"""Environment for dashboard security tests.

src.config reads ROOT_PATH and provider API key names at import time.
The values below are placeholders so the dashboard modules can be imported.
No real credential is used.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="cobbie-dashboard-tests-"))
os.environ["ROOT_PATH"] = str(_TMP)
os.environ.pop("COBBIE_ALLOW_BENCHMARK_WRITES", None)

# Names required by src/config.py. Values are placeholders, not secrets.
for _name in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "FIREWORKS_API_KEY",
    "CEREBRAS_API_KEY",
    "OPENROUTER_API_KEY",
):
    os.environ[_name] = "placeholder-not-a-secret"

_DB_DIR = _TMP / "src" / "db"
_DB_DIR.mkdir(parents=True)
DB_PATH = _DB_DIR / "db.db"

_SCHEMA = """
CREATE TABLE ifcmodels (
  project_name TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_path TEXT NOT NULL,
  model_description TEXT NOT NULL,
  id INTEGER PRIMARY KEY
);
CREATE TABLE ifc_bench (
  question TEXT NOT NULL,
  ground_truth TEXT NOT NULL,
  ifc_id INTEGER NOT NULL,
  id INTEGER PRIMARY KEY,
  category INTEGER CHECK (category BETWEEN 1 AND 4),
  cobbie TEXT
);
"""

# Two rows so load_train_dev_split places one in the train split and one in dev.
ORIGINAL_ROWS = {
    1: ("How many doors?", "There are 4 doors.", 1),
    2: ("How many windows?", "There are 8 windows.", 2),
}


def reset_benchmark_rows() -> None:
    """Restore the benchmark rows the dashboard tests edit."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM ifc_bench")
        conn.execute("DELETE FROM ifcmodels")
        conn.execute(
            "INSERT INTO ifcmodels (project_name, model_name, model_path, model_description, id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("Proj", "Model", "/tmp/m.ifc", "desc", 1),
        )
        for question_id, (question, ground_truth, category) in ORIGINAL_ROWS.items():
            conn.execute(
                "INSERT INTO ifc_bench (question, ground_truth, ifc_id, id, category) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, ground_truth, 1, question_id, category),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_benchmark_rows() -> dict[int, tuple[str, str, int]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        found = conn.execute(
            "SELECT id, question, ground_truth, category FROM ifc_bench ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return {row[0]: (row[1], row[2], row[3]) for row in found}


if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.executescript(_SCHEMA)
        connection.commit()
    finally:
        connection.close()

reset_benchmark_rows()
