"""Test process configuration.

Imported before test modules. Sets placeholder credentials and a local MLflow
store so importing the training pipeline does not require real secrets or a
running MLflow server, and ensures the SQLite schema the dataset loader reads
exists.
"""

import os
import sqlite3

os.environ.setdefault("ROOT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/cobbie-mlflow-tests")
os.environ.setdefault("BAML_LOG", "off")

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
    "Z_AI_API_KEY",
):
    os.environ.setdefault(_name, "placeholder")

_DB_PATH = os.path.join(os.environ["ROOT_PATH"], "src", "db", "db.db")
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
_schema = """
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
with sqlite3.connect(_DB_PATH) as _conn:
    _conn.executescript(_schema)
