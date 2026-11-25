"""Python execution utilities for COBBIE and other components.

Model-written code is never executed in this process: it runs in a sandboxed worker
(see src/util/sandbox.py) that has no API keys and no write access to the tools directories.
"""

from typing import Any, Callable, Dict, Iterable, Optional

import tiktoken
from loguru import logger

from src.util.sandbox import SandboxedInterpreter


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in the given text using tiktoken.

    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization (default: gpt-3.5-turbo)

    Returns:
        The number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to a default encoding if the model is not found
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens
        model: The model name to use for tokenization

    Returns:
        Truncated text with warning message if truncation occurred
    """
    token_count = count_tokens(text, model)

    if token_count <= max_tokens:
        return text

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode and truncate
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)

    warning_msg = f"\n...\nWARNING:\nOutput truncated due to excessive length ({token_count} tokens > {max_tokens} tokens limit)"

    return truncated_text + warning_msg


def setup_interpreter(
    model_path: Optional[str] = None,
    tools: Optional[Dict[str, Callable]] = None,
    readable_paths: Optional[Iterable[str]] = None,
) -> SandboxedInterpreter:
    """
    Setup a sandboxed Python session with tools and model context.

    Args:
        model_path: Optional path to IFC model file (exposed as ``path_ifc_model``)
        tools: Dictionary of available tools/functions
        readable_paths: Additional IFC model files the code may open (e.g. models for testing)

    Returns:
        A persistent sandboxed interpreter

    Raises:
        SandboxUnavailableError: if the sandbox cannot be set up on this machine
    """
    try:
        interpreter = SandboxedInterpreter(
            model_path=model_path, tools=tools, readable_paths=readable_paths
        )

        logger.debug(f"Interpreter setup with {len(tools) if tools else 0} tools")
        if model_path:
            logger.debug(f"Model path set: {model_path}")

        return interpreter

    except Exception as e:
        logger.error(f"Failed to setup interpreter: {e}")
        raise


def execute_python(
    python_code: str,
    tools: Dict[str, Callable],
    model_path: Optional[str] = None,
    max_tokens: int = 2048,
    interpreter: Optional[Any] = None,
) -> str:
    """
    Execute Python code and return output.

    Args:
        python_code: Python code to execute
        tools: Dictionary of available tools/functions
        model_path: Optional path to IFC model file
        max_tokens: Maximum number of tokens in output (default: 2048)
        interpreter: Optional existing interpreter to reuse (skips setup_interpreter if provided)

    Returns:
        String output from code execution (truncated if exceeds max_tokens)
    """
    owned_interpreter = None
    try:
        # Use provided interpreter or create a new one
        if interpreter is None:
            interpreter = owned_interpreter = setup_interpreter(model_path, tools)
        assert interpreter is not None

        execution = interpreter.run(python_code)
        if execution["status"] == "syntax_error":
            return f"Syntax Error: {execution['message']}"
        if execution["status"] != "ok":
            return f"Runtime Error: {execution['message']}"

        stdout_output = execution.get("stdout") or ""
        stderr_output = execution.get("stderr") or ""

        # Combine outputs
        if stdout_output and stderr_output:
            result = f"STDOUT:\n{stdout_output}\nSTDERR:\n{stderr_output}"
        elif stdout_output:
            result = stdout_output
        elif stderr_output:
            result = f"STDERR:\n{stderr_output}"
        else:
            result = "Code executed successfully but produced no visible output. If you called a function that returns a value, make sure to print it: result = my_function(...); print(result)"

        result = result.strip()

        # Truncate output if it exceeds max_tokens
        result = truncate_to_tokens(result, max_tokens)

        logger.debug(f"Python execution completed. Output length: {len(result)}")

        logger.info(f"Output: {result[:50]}...")
        return result

    except Exception as e:
        error_msg = f"Failed to execute Python code: {e}"
        logger.error(error_msg)
        return error_msg

    finally:
        if owned_interpreter is not None:
            owned_interpreter.close()


def execute_python_safe(
    python_code: str,
    tools: Dict[str, Callable],
    model_path: Optional[str] = None,
    timeout_seconds: int = 30,
    max_tokens: int = 2048,
    interpreter: Optional[Any] = None,
) -> str:
    """
    Execute Python code with timeout protection.

    Args:
        python_code: Python code to execute
        tools: Dictionary of available tools/functions
        model_path: Optional path to IFC model file
        timeout_seconds: Maximum execution time in seconds
        max_tokens: Maximum number of tokens in output (default: 2048)
        interpreter: Optional existing interpreter to reuse

    Returns:
        String output from code execution or timeout message
    """
    import threading

    result_container: Dict[str, Any] = {"result": None, "completed": False}

    def target():
        try:
            result_container["result"] = execute_python(python_code, tools, model_path, max_tokens, interpreter)
            result_container["completed"] = True
        except Exception as e:
            result_container["result"] = f"Execution failed: {e}"
            result_container["completed"] = True

    # Start thread
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()

    # Wait for completion or timeout
    thread.join(timeout_seconds)

    if not result_container["completed"]:
        return f"Execution timed out after {timeout_seconds} seconds"

    result = result_container["result"]
    return result if isinstance(result, str) else str(result)
