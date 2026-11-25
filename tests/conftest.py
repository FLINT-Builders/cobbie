"""Test process setup.

Provider API keys are forced to an obvious placeholder before application
modules load dotenv, so a developer .env cannot leak into test output.
"""

import os
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDER_SECRET = "placeholder-not-a-secret"

# Names read by src/config.py and by the BAML clients. Values are placeholders.
_KEY_NAMES = (
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
    "CONTEXT7_API_KEY",
)

os.environ["ROOT_PATH"] = str(REPO_ROOT)
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlflow-cobbie-injection-tests"
for _name in _KEY_NAMES:
    os.environ[_name] = PLACEHOLDER_SECRET

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

_db_path = REPO_ROOT / "src" / "db" / "db.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_connection = sqlite3.connect(_db_path)
_connection.executescript(_SCHEMA)
_connection.commit()
_connection.close()
