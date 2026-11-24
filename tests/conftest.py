"""Test bootstrap.

Writes a gitignored .env of obvious placeholders, then builds the empty
SQLite schema src.db imports at startup. Must run before any src import.
"""

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Names config.py copies out of the environment after load_dotenv, plus the
# Z.AI key the BAML clients read from the environment.
PROVIDER_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "FIREWORKS_API_KEY",
    "CEREBRAS_API_KEY",
    "OPENROUTER_API_KEY",
    "Z_AI_API_KEY",
)

# Obvious placeholders. Not credentials.
PROVIDER_KEY_VALUES = {
    name: f"placeholder-value-for-{name}" for name in PROVIDER_KEY_NAMES
}

os.environ["ROOT_PATH"] = str(ROOT)
os.environ["BAML_LOG"] = "off"
os.environ["MLFLOW_TRACKING_URI"] = "file:/tmp/cobbie-mlflow-tests"
os.environ["PYTHON_DOTENV_DISABLED"] = "false"

env_lines = [f"ROOT_PATH={ROOT}"]
for name, value in PROVIDER_KEY_VALUES.items():
    env_lines.append(f"{name}={value}")
    # Do not pre-set these. config.py must obtain them via load_dotenv.
    os.environ.pop(name, None)
(ROOT / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

_db_path = ROOT / "src" / "db" / "db.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(_db_path) as _conn:
    _conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ifcmodels (
            project_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_description TEXT NOT NULL,
            id INTEGER PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS ifc_bench (
            question TEXT NOT NULL,
            ground_truth TEXT NOT NULL,
            ifc_id INTEGER NOT NULL,
            id INTEGER PRIMARY KEY,
            category INTEGER CHECK (category BETWEEN 1 AND 4),
            cobbie TEXT,
            FOREIGN KEY (ifc_id) REFERENCES ifcmodels (id)
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
