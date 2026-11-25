"""Model-written helper functions, described from their source without executing it."""

import ast
import inspect
import textwrap
from typing import List, Optional


class _SourceText:
    """Renders an annotation or default value exactly as written in the source."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _text(node: Optional[ast.AST]):
    return inspect.Parameter.empty if node is None else _SourceText(ast.unparse(node))


def _signature_from_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[inspect.Signature]:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: List[Optional[ast.expr]] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)

    params = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        params.append(inspect.Parameter(arg.arg, kind, default=_text(default), annotation=_text(arg.annotation)))
    if args.vararg:
        params.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL, annotation=_text(args.vararg.annotation))
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY, default=_text(default), annotation=_text(arg.annotation))
        )
    if args.kwarg:
        params.append(
            inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD, annotation=_text(args.kwarg.annotation))
        )
    try:
        return inspect.Signature(params, return_annotation=_text(node.returns))
    except (TypeError, ValueError):
        return None


class GeneratedTool:
    """
    A helper function written by a model (created, enhanced or debugged during training).

    The training and evaluation processes hold provider API keys, so they never execute
    this source. ``inspect.signature``, ``inspect.getdoc`` and ``__name__`` work as for a
    regular function, which is all the prompt builders need; the code itself only runs
    inside the sandboxed interpreter (see ``src.util.sandbox``).
    """

    def __init__(self, name: str, source: str, module: Optional[str] = None):
        source = textwrap.dedent(source).strip() + "\n"
        tree = ast.parse(source)
        node = next(
            (
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
            ),
            None,
        )
        if node is None:
            raise ValueError(f"Function {name} not found in generated code")

        self.source = source
        self.module_name = module or f"cobbie_generated.{name}"
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = self.module_name
        self.__doc__ = ast.get_docstring(node) or ""
        self.__signature__ = _signature_from_ast(node)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Generated helper {self.__name__!r} can only run inside the sandboxed interpreter."
        )

    def __repr__(self) -> str:
        return f"<GeneratedTool {self.__name__} from {self.module_name}>"

    @classmethod
    def from_module_source(cls, source: str, module: str) -> List["GeneratedTool"]:
        """All public top-level functions of a generated module, in source order."""
        tree = ast.parse(source)
        return [
            cls(node.name, source, module=module)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        ]
