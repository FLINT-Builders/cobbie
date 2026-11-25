"""Environment for tests that import the training pipeline.

The training modules read ROOT_PATH and provider keys at import time, and
src.db loads the IFC-Bench tables while it is imported. Placeholder keys are
not credentials. The database file is created empty so those imports succeed.
"""

import os
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ["ROOT_PATH"] = str(REPO_ROOT)
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////tmp/cobbie-assess-injection-mlflow.db"
os.environ.setdefault("Z_AI_API_KEY", "placeholder")
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
with sqlite3.connect(_db_path) as _connection:
    _connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS ifcmodels (
            project_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_description TEXT NOT NULL,
            id INTEGER NOT NULL,
            PRIMARY KEY (id)
        );
        CREATE TABLE IF NOT EXISTS tool_usage_stats (
            tool_name TEXT NOT NULL,
            questions_when_included INTEGER DEFAULT 0,
            questions_when_called INTEGER DEFAULT 0,
            questions_correct_contribution INTEGER DEFAULT 0,
            questions_wrong_contribution INTEGER DEFAULT 0,
            created_at_question INTEGER DEFAULT 0,
            last_question_processed INTEGER DEFAULT 0,
            PRIMARY KEY (tool_name)
        );
        CREATE TABLE IF NOT EXISTS tool_usage_stats_eval (
            tool_name TEXT NOT NULL,
            questions_when_included INTEGER DEFAULT 0 NOT NULL,
            questions_when_called INTEGER DEFAULT 0 NOT NULL,
            questions_correct_contribution INTEGER DEFAULT 0 NOT NULL,
            questions_wrong_contribution INTEGER DEFAULT 0 NOT NULL,
            PRIMARY KEY (tool_name)
        );
        CREATE TABLE IF NOT EXISTS ifc_bench (
            question TEXT NOT NULL,
            ground_truth TEXT NOT NULL,
            ifc_id INTEGER NOT NULL,
            id INTEGER NOT NULL,
            category INTEGER,
            cobbie TEXT,
            PRIMARY KEY (id),
            CHECK (category BETWEEN 1 AND 4),
            FOREIGN KEY(ifc_id) REFERENCES ifcmodels (id)
        );
        """
    )
