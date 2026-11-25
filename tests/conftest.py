"""Environment required before any Cobbie module import.

src.config reads ROOT_PATH and provider keys at import time. The training
script also opens the local SQLite dataset while it is imported. Neither
value is a credential; the keys are placeholders because these tests never
call a model API.
"""

import os
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("ROOT_PATH", str(REPO_ROOT))
os.environ.setdefault("BAML_LOG", "off")
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/cobbie-mlflow")
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

_db_path = REPO_ROOT / "src" / "db" / "db.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_con = sqlite3.connect(_db_path)
_con.executescript(
    """
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
        ifc_id INTEGER NOT NULL,
        category INTEGER,
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
)
_con.commit()
_con.close()
