"""BAML client used by the project.

The client generated from baml_src lives in src/baml/generated/baml_client
(regenerate with ``cd src/baml && uv run baml-cli generate``). This package
re-exports it with ``b`` wrapped so that arguments carrying execution output or
IFC-derived history are framed as untrusted data before a prompt renders them
(see src/baml/untrusted.py). Import from here, not from the generated package.
"""

from src.baml.generated.baml_client import (
    config,
    partial_types,
    reset_baml_env_vars,
    stream_types,
    tracing,
    types,
    watchers,
)
from src.baml.generated.baml_client import b as _generated_b
from src.baml.untrusted import framed_client

b = framed_client(_generated_b)

__all__ = [
    "b",
    "stream_types",
    "partial_types",
    "tracing",
    "types",
    "reset_baml_env_vars",
    "config",
    "watchers",
]
