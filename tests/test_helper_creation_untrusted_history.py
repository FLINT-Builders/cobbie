"""Security spec for unsanitized execution history in helper creation.

These tests fail while the vulnerability is present. They pass when:

* The creator, debugger, and question-answering prompts place execution
  history, the question, the reference answer, any existing implementation,
  the debugger error description, and previous code-step output in blocks
  delimited by the exact sentinels ``<<<UNTRUSTED_DATA`` and
  ``UNTRUSTED_DATA>>>``.
* The same prompts tell the model the blocks are untrusted, to treat them
  only as data, and not to follow them as instructions.
* Python returned as a code action by helper creation, helper debugging, and
  the question-answering loop cannot read provider API keys from the
  environment and cannot create a file in the tools directory. That must be
  true of the code step itself, which runs before any keep-or-discard
  decision.

The live model call is replaced with a scripted code action. Prompt rendering
uses the generated BAML request builder, and execution uses the product's
code-action loop.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("ROOT_PATH", str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:/tmp/mlflow-cobbie-tests")
os.environ.setdefault("BAML_LOG", "off")
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
    "Z_AI_API_KEY",
):
    os.environ.setdefault(_key, "placeholder-not-a-secret")

_DB = Path(os.environ["ROOT_PATH"]) / "src" / "db" / "db.db"
_DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(_DB) as _con:
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

import ifcopenshell  # noqa: E402
import pytest  # noqa: E402

from src.baml.baml_client import b  # noqa: E402
from src.baml.baml_client.types import (  # noqa: E402
    AnswerEvaluationResult,
    CodeAction,
    FaultyToolAnalysis,
    FinalAnswer,
    HelperFunctionAssessment,
    NewHelperFunction,
    NewToolAnalysis,
    UpdatedHelperFunction,
)
from typing import Callable, Dict, Literal, Optional  # noqa: E402

from baml_py.baml_py import Collector  # noqa: E402
from src.config import CREATED_TOOLS_PATH  # noqa: E402
from src.db.models import IfcBench, Ifcmodels  # noqa: E402
from src.schemas.training_context import Context  # noqa: E402

_TRAINING_PATH = Path(os.environ["ROOT_PATH"]) / "scripts" / "run_training_phase.py"
_spec = importlib.util.spec_from_file_location("run_training_phase", _TRAINING_PATH)
training = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(training)

# CobbiResult annotates its answer with FinalAnswer, which is imported only
# for type checking. Pydantic 2.12.5 (the locked version) then refuses to
# build Context until that name is supplied. The training command hits this
# at Context(...) and the exception is caught, so the tests rebuild the model
# and call the handlers directly.
import src.schemas.agent_error as _agent_error

_agent_error.FinalAnswer = FinalAnswer
Context.model_rebuild(
    _types_namespace={
        "FinalAnswer": FinalAnswer,
        "Collector": Collector,
        "CobbiResult": _agent_error.CobbiResult,
        "AnswerEvaluationResult": AnswerEvaluationResult,
        "NewToolAnalysis": NewToolAnalysis,
        "NewHelperFunction": NewHelperFunction,
        "FaultyToolAnalysis": FaultyToolAnalysis,
        "UpdatedHelperFunction": UpdatedHelperFunction,
        "HelperFunctionAssessment": HelperFunctionAssessment,
        "IfcBench": IfcBench,
        "Optional": Optional,
        "Dict": Dict,
        "Callable": Callable,
        "Literal": Literal,
    }
)

BEGIN = "<<<UNTRUSTED_DATA"
END = "UNTRUSTED_DATA>>>"
NAME_MARKER = "NAME_MARKER_8161"
PROP_MARKER = "PROP_MARKER_8161"
QUESTION_MARKER = "QUESTION_MARKER_8161"
ANSWER_MARKER = "ANSWER_MARKER_8161"
IMPL_MARKER = "IMPL_MARKER_8161"
ERROR_MARKER = "ERROR_MARKER_8161"
ROUNDTRIP = "ROUNDTRIP_MARKER_8161"
PROBE_NAME = "_security_probe_do_not_keep.py"


def _write_ifc(directory: Path) -> Path:
    path = directory / "attacker.ifc"
    model = ifcopenshell.file(schema="IFC4")
    model.create_entity(
        "IfcWall",
        GlobalId=ifcopenshell.guid.new(),
        Name=NAME_MARKER,
        Description="wall",
    )
    model.create_entity(
        "IfcPropertySingleValue",
        Name="MarkerProp",
        NominalValue=model.create_entity("IfcText", PROP_MARKER),
    )
    model.write(str(path))
    return path


def _print_ifc_code() -> str:
    return (
        "model = ifcopenshell.open(path_ifc_model)\n"
        "for el in model.by_type('IfcWall'):\n"
        "    print(el.Name)\n"
        "for prop in model.by_type('IfcPropertySingleValue'):\n"
        "    nominal = prop.NominalValue\n"
        "    print(nominal.wrappedValue if nominal is not None else '')\n"
    )


def _probe_code() -> str:
    probe = Path(CREATED_TOOLS_PATH) / PROBE_NAME
    return (
        "import os\n"
        "from pathlib import Path\n"
        f"print({ROUNDTRIP!r})\n"
        f"probe = Path({str(probe)!r})\n"
        "wrote = False\n"
        "try:\n"
        "    probe.write_text('probe')\n"
        "    wrote = probe.exists()\n"
        "except Exception:\n"
        "    wrote = False\n"
        "print('WROTE_TOOL' if wrote else 'NOWRITE_TOOL')\n"
        "print('SAW_KEY' if os.environ.get('Z_AI_API_KEY') else 'NO_KEY')\n"
    )


def _assert_untrusted(prompt: str, payload: str) -> None:
    lowered = prompt.lower()
    assert "untrusted" in lowered
    assert "only as data" in lowered
    assert "not" in lowered and "instruction" in lowered
    assert payload in prompt, "attacker-controlled text did not reach the prompt"
    start = 0
    while True:
        index = prompt.find(payload, start)
        if index == -1:
            break
        begin = prompt.rfind(BEGIN, 0, index)
        end = prompt.find(END, index + len(payload))
        assert begin != -1 and end != -1, (
            "attacker-controlled text is outside an untrusted-data block"
        )
        start = index + len(payload)


def _prompt_text(request) -> str:
    body = request.body
    if hasattr(body, "text"):
        return body.text()
    raw = body.raw()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


class _ScriptedModel:
    def __init__(self, scripts: dict[str, list], prompts: dict[str, list[str]]):
        self._scripts = {name: list(values) for name, values in scripts.items()}
        self._prompts = prompts

    def _record(self, name: str, request_fn, kwargs: dict):
        self._prompts.setdefault(name, []).append(_prompt_text(request_fn(**kwargs)))
        pending = self._scripts.get(name) or []
        if not pending:
            raise AssertionError(f"no scripted response left for {name}")
        return pending.pop(0)

    def Cobbie(self, **kwargs):
        return self._record("Cobbie", b.request.Cobbie, kwargs)

    def HelperFunctionCreator(self, **kwargs):
        return self._record(
            "HelperFunctionCreator", b.request.HelperFunctionCreator, kwargs
        )

    def HelperFunctionDebugger(self, **kwargs):
        return self._record(
            "HelperFunctionDebugger", b.request.HelperFunctionDebugger, kwargs
        )


@pytest.fixture
def scripted_model(monkeypatch):
    prompts: dict[str, list[str]] = {}
    holder: dict[str, _ScriptedModel] = {}

    def with_options(*_args, **_kwargs):
        return holder["model"]

    holder["model"] = _ScriptedModel({}, prompts)
    monkeypatch.setattr(b, "with_options", with_options)
    return holder, prompts


@pytest.fixture
def attacker_ifc(tmp_path: Path) -> Path:
    return _write_ifc(tmp_path)


@pytest.fixture(autouse=True)
def _remove_probe():
    probe = Path(CREATED_TOOLS_PATH) / PROBE_NAME
    probe.unlink(missing_ok=True)
    yield
    probe.unlink(missing_ok=True)


def _qa_pair(ifc_path: Path, question: str, answer: str) -> IfcBench:
    model = Ifcmodels(
        id=1,
        project_name="attacker",
        model_name="attacker",
        model_path=str(ifc_path),
        model_description="supplied model",
    )
    pair = IfcBench(
        id=1,
        question=question,
        ground_truth=answer,
        ifc_id=1,
        category=1,
    )
    pair.ifc = model
    return pair


def _run_cobbie_until_answer(ifc_path: Path, question: str):
    from src.agents.cobbie import cobbie

    return cobbie(
        user_input=question,
        tools={},
        max_iterations=4,
        model_path=str(ifc_path),
        client="GLM_4_7",
    )


def test_creator_treats_ifc_history_as_data_and_sandboxes_code(
    scripted_model, attacker_ifc: Path
):
    holder, prompts = scripted_model
    holder["model"] = _ScriptedModel(
        {
            "Cobbie": [
                CodeAction(thoughts="read the model", python_code=_print_ifc_code()),
                FinalAnswer(thoughts="done", answer="counted"),
            ],
            "HelperFunctionCreator": [
                CodeAction(thoughts="run tests", python_code=_probe_code()),
                NewHelperFunction(
                    thoughts="done",
                    function_implementation="def probe_helper():\n    return []\n",
                    success=True,
                ),
            ],
        },
        prompts,
    )

    impl_path = Path(CREATED_TOOLS_PATH) / "probe_helper.py"
    impl_path.write_text(f"# {IMPL_MARKER}\ndef probe_helper():\n    return []\n")
    try:
        result = _run_cobbie_until_answer(attacker_ifc, QUESTION_MARKER)
        assert result.answer is not None
        assert NAME_MARKER in result.history
        assert PROP_MARKER in result.history

        context = Context(
            qa_pair=_qa_pair(attacker_ifc, QUESTION_MARKER, ANSWER_MARKER),
            global_question_num=0,
            cobbie_result=result,
            identify_tool_result=NewToolAnalysis(
                thoughts="extract",
                action="enhance_existing",
                tool_name="probe_helper",
                tool_description="group elements",
                existing_tool_for_enhancement="probe_helper",
            ),
            tool_name="probe_helper",
            is_enhancement=True,
        )
        state, context = training.handle_create_new_tool(context)
    finally:
        impl_path.unlink(missing_ok=True)

    assert state == training.TrainingState.TEST_TOOL_WITH_COBBIE
    assert context.tool_saved is False
    creator_prompts = prompts["HelperFunctionCreator"]
    assert len(creator_prompts) >= 2
    for payload in (
        NAME_MARKER,
        PROP_MARKER,
        QUESTION_MARKER,
        ANSWER_MARKER,
        IMPL_MARKER,
    ):
        _assert_untrusted(creator_prompts[0], payload)
    _assert_untrusted(creator_prompts[1], ROUNDTRIP)
    combined = "\n".join(creator_prompts)
    assert "SAW_KEY" not in combined
    assert "WROTE_TOOL" not in combined
    assert not (Path(CREATED_TOOLS_PATH) / PROBE_NAME).exists()


def test_debugger_treats_ifc_history_as_data_and_sandboxes_code(
    scripted_model, attacker_ifc: Path
):
    holder, prompts = scripted_model
    holder["model"] = _ScriptedModel(
        {
            "Cobbie": [
                CodeAction(thoughts="read the model", python_code=_print_ifc_code()),
                FinalAnswer(thoughts="done", answer="counted"),
            ],
            "HelperFunctionDebugger": [
                CodeAction(thoughts="run tests", python_code=_probe_code()),
                UpdatedHelperFunction(
                    thoughts="done",
                    fixed_implementation="def probe_helper():\n    return []\n",
                    changes_summary="none",
                    success=True,
                    test_cases_provided="none",
                ),
            ],
        },
        prompts,
    )

    impl_path = Path(CREATED_TOOLS_PATH) / "probe_helper.py"
    impl_path.write_text(f"# {IMPL_MARKER}\ndef probe_helper():\n    return []\n")
    try:
        result = _run_cobbie_until_answer(attacker_ifc, QUESTION_MARKER)
        context = Context(
            qa_pair=_qa_pair(attacker_ifc, QUESTION_MARKER, ANSWER_MARKER),
            global_question_num=0,
            cobbie_result=result,
            identify_faulty_result=FaultyToolAnalysis(
                thoughts="fault",
                faulty_tool=True,
                faulty_tool_name="probe_helper",
                error_description=ERROR_MARKER,
                confidence="high",
            ),
            tool_name="probe_helper",
        )
        state, context = training.handle_debug_faulty_tool(context)
    finally:
        impl_path.unlink(missing_ok=True)

    assert state == training.TrainingState.TEST_TOOL_WITH_COBBIE
    assert context.tool_saved is False
    debug_prompts = prompts["HelperFunctionDebugger"]
    assert len(debug_prompts) >= 2
    for payload in (NAME_MARKER, PROP_MARKER, IMPL_MARKER, ERROR_MARKER):
        _assert_untrusted(debug_prompts[0], payload)
    _assert_untrusted(debug_prompts[1], ROUNDTRIP)
    combined = "\n".join(debug_prompts)
    assert "SAW_KEY" not in combined
    assert "WROTE_TOOL" not in combined
    assert not (Path(CREATED_TOOLS_PATH) / PROBE_NAME).exists()


def test_qa_loop_treats_printed_ifc_text_as_data_and_sandboxes_later_code(
    scripted_model, attacker_ifc: Path
):
    holder, prompts = scripted_model
    holder["model"] = _ScriptedModel(
        {
            "Cobbie": [
                CodeAction(thoughts="read the model", python_code=_print_ifc_code()),
                CodeAction(thoughts="follow up", python_code=_probe_code()),
                FinalAnswer(thoughts="done", answer="counted"),
            ]
        },
        prompts,
    )

    result = _run_cobbie_until_answer(attacker_ifc, QUESTION_MARKER)
    assert result.answer is not None
    cobbie_prompts = prompts["Cobbie"]
    assert len(cobbie_prompts) >= 2
    for payload in (NAME_MARKER, PROP_MARKER):
        _assert_untrusted(cobbie_prompts[1], payload)
    _assert_untrusted(cobbie_prompts[2], ROUNDTRIP)
    combined = "\n".join(cobbie_prompts)
    assert "SAW_KEY" not in combined
    assert "WROTE_TOOL" not in combined
    assert not (Path(CREATED_TOOLS_PATH) / PROBE_NAME).exists()
