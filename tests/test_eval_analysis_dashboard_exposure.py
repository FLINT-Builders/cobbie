"""Security regression tests for the evaluation analysis dashboard.

These tests are the specification for the fix. They fail while the
vulnerability is present.

Required behavior:

1. The launch command documented in README.md (``uv run streamlit run
   scripts/eval_analysis_app.py``) must listen only on a loopback address.
   Either of these satisfies that:
   - a repo ``.streamlit/config.toml`` (or ``scripts/.streamlit/config.toml``)
     with ``server.address`` set to ``127.0.0.1`` or ``localhost``
   - ``--server.address 127.0.0.1`` (or ``localhost``) on that README command

2. ``Apply Updates`` must not modify ``ifc_bench`` unless the environment
   variable ``COBBIE_ALLOW_BENCHMARK_WRITES`` is set to the exact value ``1``.
   Read that variable when the button handler runs. Without it, ``question``,
   ``ground_truth``, and ``category`` stay as they were for every question id,
   including the values ``load_train_dev_split`` returns to the evaluation and
   training judge.

The only database write in the dashboard is ``Apply Updates`` in the Single
Run Analysis tab, which calls ``update_ifc_bench_rows``. The comparison tabs
render ``st.dataframe`` and do not write.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.conftest import ORIGINAL_ROWS, fetch_benchmark_rows, reset_benchmark_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
APP_PATH = REPO_ROOT / "scripts" / "eval_analysis_app.py"
BENCHMARK_WRITE_OPT_IN = "COBBIE_ALLOW_BENCHMARK_WRITES"
OPT_IN_VALUE = "1"


def _documented_streamlit_args() -> list[str]:
    """Arguments after ``streamlit run`` on the README launch line."""
    launch_line = None
    for line in README_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "eval_analysis_app.py" in stripped and "streamlit" in stripped:
            launch_line = stripped
            break
    if launch_line is None:
        pytest.fail(f"{README_PATH} does not document how to launch {APP_PATH.name}")

    import shlex

    tokens = shlex.split(launch_line)
    if "streamlit" not in tokens:
        pytest.fail(f"could not parse streamlit command from README line: {launch_line}")
    streamlit_at = tokens.index("streamlit")
    args = tokens[streamlit_at + 1 :]
    if not args or args[0] != "run":
        pytest.fail(f"README streamlit command is not 'streamlit run': {launch_line}")
    return args[1:]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _proc_socket_inodes(pid: int) -> set[str]:
    inodes: set[str] = set()
    fd_dir = Path(f"/proc/{pid}/fd")
    if not fd_dir.is_dir():
        return inodes
    for entry in fd_dir.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[len("socket:[") : -1])
    return inodes


def _decode_proc_ip(hex_ip: str) -> str:
    if len(hex_ip) == 8:
        raw = bytes.fromhex(hex_ip)[::-1]
        return socket.inet_ntop(socket.AF_INET, raw)
    groups = [hex_ip[index : index + 8] for index in range(0, 32, 8)]
    raw = b"".join(bytes.fromhex(group)[::-1] for group in groups)
    return socket.inet_ntop(socket.AF_INET6, raw)


def _listening_addresses(pid: int, port: int) -> list[str]:
    """Local addresses of LISTEN sockets owned by ``pid`` on ``port``."""
    inodes = _proc_socket_inodes(pid)
    hex_port = f"{port:04X}"
    found: list[str] = []
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            local, state, inode = fields[1], fields[3], fields[9]
            local_ip, local_port = local.split(":")
            if state != "0A" or local_port.upper() != hex_port or inode not in inodes:
                continue
            found.append(_decode_proc_ip(local_ip))
    return found


def _is_loopback(address: str) -> bool:
    import ipaddress

    # Wildcard binds accept traffic on every interface, including the network.
    if address in {"0.0.0.0", "::"}:
        return False
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def _non_loopback_ipv4_addresses() -> list[str]:
    import ipaddress

    completed = subprocess.run(
        ["hostname", "-I"],
        check=False,
        capture_output=True,
        text=True,
    )
    addresses: list[str] = []
    for token in completed.stdout.split():
        try:
            ip = ipaddress.ip_address(token)
        except ValueError:
            continue
        if ip.version == 4 and not ip.is_loopback:
            addresses.append(token)
    return addresses


def _http_status(address: str, port: int) -> int | None:
    """Return the HTTP status from a TCP client, or None if the connect fails."""
    try:
        with socket.create_connection((address, port), timeout=2) as sock:
            request = (
                f"GET / HTTP/1.1\r\nHost: {address}:{port}\r\nConnection: close\r\n\r\n"
            )
            sock.sendall(request.encode())
            payload = b""
            while b"\r\n" not in payload:
                chunk = sock.recv(256)
                if not chunk:
                    break
                payload += chunk
    except OSError:
        return None
    first = payload.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
    parts = first.split()
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def test_documented_dashboard_is_not_reachable_off_loopback() -> None:
    """A client on another interface must not be able to open the dashboard.

    The README command is launched with no server address other than flags
    that command already carries. Headless mode and an ephemeral port are
    test harness settings; they do not set server.address.
    """
    documented_args = _documented_streamlit_args()
    port = _free_port()
    command = [sys.executable, "-m", "streamlit", "run", *documented_args]
    if not any(arg == "--server.port" or arg.startswith("--server.port=") for arg in command):
        command.extend(["--server.port", str(port)])
    else:
        # The documented command pinned a port. Observe that port.
        if "--server.port" in command:
            port = int(command[command.index("--server.port") + 1])
        else:
            port_arg = next(arg for arg in command if arg.startswith("--server.port="))
            port = int(port_arg.split("=", 1)[1])
    if not any(
        arg == "--server.headless" or arg.startswith("--server.headless=") for arg in command
    ):
        command.extend(["--server.headless", "true"])

    home = Path(os.environ["ROOT_PATH"]) / "streamlit-home"
    home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env.pop("STREAMLIT_SERVER_ADDRESS", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")]
    )

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        addresses: list[str] = []
        deadline = time.time() + 20
        assert process.stdout is not None
        while time.time() < deadline:
            addresses = _listening_addresses(process.pid, port)
            if addresses:
                break
            if process.poll() is not None:
                output = process.stdout.read()
                pytest.fail(
                    "dashboard process exited before it listened\n"
                    + textwrap.shorten(output, width=2000, placeholder=" ...")
                )
            time.sleep(0.1)
        assert addresses, "dashboard did not open a listening socket"
        non_loopback = [address for address in addresses if not _is_loopback(address)]
        assert not non_loopback, (
            "documented dashboard launch is reachable off loopback; "
            f"listening addresses: {addresses}"
        )

        for address in _non_loopback_ipv4_addresses():
            status = _http_status(address, port)
            assert status is None, (
                f"non-loopback client {address}:{port} received HTTP {status} "
                "from the dashboard with no credentials"
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _judge_inputs(row: object) -> tuple[str, str, int | None]:
    """The question, ground truth, and category the eval judge receives.

    This is the same attribute lookup as scripts/run_evaluation.py
    process_question and scripts/run_training_phase.py handle_verify_answer.
    """
    question = getattr(row, "question", "")
    ground_truth = getattr(row, "answer", "") or getattr(row, "ground_truth", "")
    category = getattr(row, "category", None)
    return question, ground_truth, category


def _apply_benchmark_edit(question_id: int, question: str, ground_truth: str, category: int):
    """Drive the dashboard script the way the data editor submits one row.

    The widget payload is the JSON shape Streamlit's data editor sends:
    ``edited_rows`` maps a row position to column names the editor left
    enabled (``update``, ``question``, ``ground_truth``, ``category``).
    """
    import mlflow

    import src.config as config
    from streamlit.proto.WidgetStates_pb2 import WidgetStates
    from streamlit.testing.v1 import AppTest

    tracking_uri = f"sqlite:///{os.environ['ROOT_PATH']}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Evaluation")
    experiment = mlflow.get_experiment_by_name("Evaluation")
    assert experiment is not None
    existing = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if existing.empty:
        with mlflow.start_run(run_name="parent-eval"):
            for qid in ORIGINAL_ROWS:
                with mlflow.start_run(run_name=f"question-{qid}", nested=True):
                    mlflow.log_params(
                        {
                            "question_id": str(qid),
                            "classification": "wrong",
                            "faithfulness": "No",
                            "completeness": "No",
                            "transparency": "No",
                            "relevance": "No",
                            "answer": "a wrong answer",
                            "justification": "nope",
                        }
                    )
                    mlflow.log_metric("cobbie_calls_count", 1)

    config.MLFLOW_URI = tracking_uri
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    assert not app.exception, app.exception

    editors = [frame for frame in app.dataframe if "update" in list(frame.value.columns)]
    assert len(editors) == 1, "Single Run Analysis did not render the benchmark editor"
    editor = editors[0]
    # The whole-frame disabled flag is what the frontend treats as read-only.
    # A column list leaves this false and only disables the named columns.
    assert editor.proto.disabled is False

    shown = editor.value
    matches = shown.index[shown["question_id"] == question_id]
    assert len(matches) == 1
    row_position = int(matches[0])
    payload = {
        "edited_rows": {
            str(row_position): {
                "update": True,
                "question": question,
                "ground_truth": ground_truth,
                "category": category,
            }
        },
        "added_rows": [],
        "deleted_rows": [],
    }
    widget_states = WidgetStates()
    edited = widget_states.widgets.add()
    edited.id = editor.proto.id
    edited.string_value = json.dumps(payload)
    buttons = [button for button in app.button if button.label == "Apply Updates"]
    assert len(buttons) == 1
    clicked = widget_states.widgets.add()
    clicked.id = buttons[0].id
    clicked.trigger_value = True
    app._run(widget_states)
    assert not app.exception, app.exception
    return app


def test_unauthenticated_apply_updates_does_not_rewrite_benchmark_or_judge_inputs() -> None:
    """Without the write opt-in, Apply Updates must leave the benchmark intact."""
    os.environ.pop(BENCHMARK_WRITE_OPT_IN, None)
    reset_benchmark_rows()

    app = _apply_benchmark_edit(
        question_id=1,
        question="tampered question",
        ground_truth="tampered ground truth",
        category=4,
    )

    assert fetch_benchmark_rows() == ORIGINAL_ROWS
    assert not any("Updated" in message.value for message in app.success)

    from src.db.load_dataset import load_train_dev_split

    train_rows, dev_rows = load_train_dev_split()
    assert train_rows and dev_rows
    for row in [*train_rows, *dev_rows]:
        question, ground_truth, category = _judge_inputs(row)
        original_question, original_truth, original_category = ORIGINAL_ROWS[row.id]
        assert question == original_question
        assert ground_truth == original_truth
        assert category == original_category


def test_benchmark_write_opt_in_updates_only_the_selected_question() -> None:
    """With the opt-in set, only the ticked question id is rewritten."""
    os.environ[BENCHMARK_WRITE_OPT_IN] = OPT_IN_VALUE
    try:
        reset_benchmark_rows()
        app = _apply_benchmark_edit(
            question_id=1,
            question="revised question",
            ground_truth="revised ground truth",
            category=3,
        )
        assert any("Updated" in message.value for message in app.success)
        stored = fetch_benchmark_rows()
        assert stored[1] == ("revised question", "revised ground truth", 3)
        assert stored[2] == ORIGINAL_ROWS[2]

        from src.db.load_dataset import load_train_dev_split

        train_rows, dev_rows = load_train_dev_split()
        by_id = {row.id: _judge_inputs(row) for row in [*train_rows, *dev_rows]}
        assert by_id[1] == ("revised question", "revised ground truth", 3)
        assert by_id[2] == ORIGINAL_ROWS[2]
    finally:
        os.environ.pop(BENCHMARK_WRITE_OPT_IN, None)
        reset_benchmark_rows()
