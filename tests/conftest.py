"""Environment bootstrap for path-confinement tests.

``src.config`` reads ``ROOT_PATH`` and provider key names at import time.
The values are placeholders so the modules can load; the tests never call
a provider. The SQLite file is empty aside from the tables import-time
dataset loading expects.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="cobbie-path-tests-"))
(_ROOT / "src" / "db").mkdir(parents=True)
(_ROOT / "src" / "tools" / "created").mkdir(parents=True)
(_ROOT / "src" / "db" / "bim_models").mkdir(parents=True)

os.environ["ROOT_PATH"] = str(_ROOT)
for _key in (
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
    os.environ.setdefault(_key, "placeholder")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ifcmodels (
    id INTEGER PRIMARY KEY,
    project_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_path TEXT NOT NULL,
    model_description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ifc_bench (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    ifc_id INTEGER NOT NULL REFERENCES ifcmodels(id),
    category INTEGER CHECK (category BETWEEN 1 AND 4),
    cobbie TEXT
);
CREATE TABLE IF NOT EXISTS tool_usage_stats (
    tool_name TEXT PRIMARY KEY,
    questions_when_included INTEGER DEFAULT 0,
    questions_when_called INTEGER DEFAULT 0,
    questions_correct_contribution INTEGER DEFAULT 0,
    questions_wrong_contribution INTEGER DEFAULT 0,
    created_at_question INTEGER DEFAULT 0,
    last_question_processed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tool_usage_stats_eval (
    tool_name TEXT PRIMARY KEY,
    questions_when_included INTEGER NOT NULL DEFAULT 0,
    questions_when_called INTEGER NOT NULL DEFAULT 0,
    questions_correct_contribution INTEGER NOT NULL DEFAULT 0,
    questions_wrong_contribution INTEGER NOT NULL DEFAULT 0
);
"""

_db_path = _ROOT / "src" / "db" / "db.db"
with sqlite3.connect(_db_path) as _conn:
    _conn.executescript(_SCHEMA)
    _conn.commit()
