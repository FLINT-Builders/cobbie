"""Frame untrusted text before it reaches a BAML prompt.

Execution output contains whatever the IFC model holds (element names,
descriptions, property values), so it can carry instructions written by
whoever authored the model. Every BAML argument that carries execution output,
or a history built from it, is base64-encoded inside an
``<untrusted-model-content>`` block before the prompt template renders it. The
base64 alphabet has no ``<``, so the content cannot close the block, and the
prompts tell the model that the block is data, not instructions
(``UntrustedContentNotice`` in baml_src/untrusted.baml).

BAML templates have no base64 filter, so the encoding is applied to the call
arguments by wrapping the generated client.
"""

import base64
import functools
import inspect
from typing import Any, Callable, Dict, Tuple

OPEN_TAG = '<untrusted-model-content encoding="base64">'
CLOSE_TAG = "</untrusted-model-content>"

# BAML function name -> parameters that carry execution output or history.
UNTRUSTED_ARGUMENTS: Dict[str, Tuple[str, ...]] = {
    "Cobbie": ("previous_attempts",),
    "HelperFunctionCreator": ("history", "previous_attempts"),
    "HelperFunctionDebugger": ("history_faulty_tool_use", "previous_attempts"),
    "HelperFunctionIdentifier": ("history",),
    "FaultyNewToolAnalysis": ("history",),
    "HelperFunctionAssessor": ("execution_history",),
    "StaticAnswerSynthesizer": ("execution_output",),
}


def frame_untrusted(text: str) -> str:
    """Return ``text`` base64-encoded inside an untrusted-content block."""
    payload = base64.b64encode(text.encode("utf-8", errors="replace")).decode("ascii")
    return f"{OPEN_TAG}\n{payload}\n{CLOSE_TAG}"


def _framed_function(function: Callable, parameters: Tuple[str, ...]) -> Callable:
    signature = inspect.signature(function)

    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        for name in parameters:
            value = bound.arguments.get(name)
            if isinstance(value, str) and value:
                bound.arguments[name] = frame_untrusted(value)
        return function(*bound.args, **bound.kwargs)

    return call


class FramedClient:
    """Proxy for a generated BAML client that frames untrusted arguments.

    Covers direct calls (``b.Cobbie``), ``b.with_options(...)``, and the
    ``request``/``stream``/``stream_request`` sub-clients.
    """

    def __init__(self, inner: Any, client_modules: frozenset):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_client_modules", client_modules)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._inner, name)
        parameters = UNTRUSTED_ARGUMENTS.get(name)
        if parameters is not None and callable(value):
            return _framed_function(value, parameters)
        if name == "with_options":

            @functools.wraps(value)
            def with_options(*args: Any, **kwargs: Any) -> "FramedClient":
                return FramedClient(value(*args, **kwargs), self._client_modules)

            return with_options
        if type(value).__module__ in self._client_modules:
            return FramedClient(value, self._client_modules)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FramedClient is read-only")

    def __repr__(self) -> str:
        return f"FramedClient({self._inner!r})"


def framed_client(client: Any) -> FramedClient:
    package = type(client).__module__.rsplit(".", 1)[0]
    return FramedClient(
        client, frozenset({f"{package}.sync_client", f"{package}.async_client"})
    )
