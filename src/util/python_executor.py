"""Python execution utilities for COBBIE and other components.

Generated code runs in a sandboxed worker process (see src/util/sandbox.py),
never in this process.
"""

from typing import Any, Callable, Dict, Iterable, Optional

import tiktoken
from loguru import logger

from src.util.sandbox import ExecutionResult, SandboxedInterpreter


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
    Start a persistent sandboxed Python session with tools and model context.

    Args:
        model_path: Optional path to IFC model file (readable inside the sandbox)
        tools: Dictionary of available tools/functions
        readable_paths: Other IFC model files the code may open

    Returns:
        A SandboxedInterpreter; raises SandboxUnavailableError if the sandbox
        cannot be established.
    """
    try:
        interpreter = SandboxedInterpreter(model_path, tools, readable_paths=readable_paths)

        logger.debug(f"Interpreter setup with {len(tools) if tools else 0} tools")
        if model_path:
            logger.debug(f"Model path set: {model_path}")

        return interpreter

    except Exception as e:
        logger.error(f"Failed to setup interpreter: {e}")
        raise


def _format_result(result: ExecutionResult, max_tokens: int) -> str:
    if result.syntax_error is not None:
        return f"Syntax Error: {result.syntax_error}"
    if result.runtime_error is not None:
        return f"Runtime Error: {result.runtime_error}"
    if result.timed_out_after is not None:
        output = f"Execution timed out after {result.timed_out_after:g} seconds"
    elif result.failure is not None:
        output = f"Failed to execute Python code: {result.failure}"
    elif result.stdout and result.stderr:
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    elif result.stdout:
        output = result.stdout
    elif result.stderr:
        output = f"STDERR:\n{result.stderr}"
    else:
        output = "Code executed successfully but produced no visible output. If you called a function that returns a value, make sure to print it: result = my_function(...); print(result)"

    if result.notice:
        output = f"{result.notice}\n{output}"

    return truncate_to_tokens(output.strip(), max_tokens)


def _execute(
    python_code: str,
    tools: Dict[str, Callable],
    model_path: Optional[str],
    max_tokens: int,
    interpreter: Optional[Any],
    timeout_seconds: Optional[float],
) -> str:
    try:
        owned = interpreter is None
        if owned:
            interpreter = setup_interpreter(model_path, tools)
        assert isinstance(interpreter, SandboxedInterpreter)
        try:
            result = _format_result(interpreter.run(python_code, timeout_seconds), max_tokens)
        finally:
            if owned:
                interpreter.close()

        logger.debug(f"Python execution completed. Output length: {len(result)}")

        logger.info(f"Output: {result[:50]}...")
        return result

    except Exception as e:
        error_msg = f"Failed to execute Python code: {e}"
        logger.error(error_msg)
        return error_msg


def execute_python(
    python_code: str,
    tools: Dict[str, Callable],
    model_path: Optional[str] = None,
    max_tokens: int = 2048,
    interpreter: Optional[Any] = None,
) -> str:
    """
    Execute Python code in the sandbox and return output.

    Args:
        python_code: Python code to execute
        tools: Dictionary of available tools/functions
        model_path: Optional path to IFC model file
        max_tokens: Maximum number of tokens in output (default: 2048)
        interpreter: Optional existing interpreter to reuse (skips setup_interpreter if provided)

    Returns:
        String output from code execution (truncated if exceeds max_tokens)
    """
    return _execute(python_code, tools, model_path, max_tokens, interpreter, None)


def execute_python_safe(
    python_code: str,
    tools: Dict[str, Callable],
    model_path: Optional[str] = None,
    timeout_seconds: int = 30,
    max_tokens: int = 2048,
    interpreter: Optional[Any] = None,
) -> str:
    """
    Execute Python code in the sandbox with a shorter timeout.

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
    return _execute(python_code, tools, model_path, max_tokens, interpreter, timeout_seconds)
