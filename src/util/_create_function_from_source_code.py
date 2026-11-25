from typing import Callable

from src.schemas import Result, ok, err
from src.util.generated_tool import GeneratedTool


def _create_function_from_source_code(
    code: str,
    function_name: str,
) -> Result[Callable, str]:
    """
    Create a tool from generated source code without executing it in this process.

    The returned tool exposes the function's name, signature and docstring, so it can be
    documented in prompts and handed to Cobbie. Its code runs only inside the sandboxed
    interpreter, where ifcopenshell (with its util submodules), math, json, typing, numpy
    and pandas are pre-imported for it.

    Args:
        code: The source code containing the function to create
        function_name: The name of the function to extract from the code

    Returns:
        Result[Callable, str]: Ok(tool) on success, Err(error_message) on failure.
                              Use .is_ok() and .is_err() to check the result status.
    """
    try:
        return ok(GeneratedTool(function_name, code))
    except Exception as e:
        return err(str(e))


if __name__ == "__main__":
    python_code = '''import ifcopenshell
from typing import List

def get_emergency_exit_doors(path_ifc_model: str) -> List:
    """Retrieves a list of IfcDoor entities that are designated as emergency exits."""
    return ifcopenshell.open(path_ifc_model).by_type('IfcDoor')'''

    creation_result = _create_function_from_source_code(
        function_name="get_emergency_exit_doors", code=python_code
    )

    if creation_result.is_ok():
        import inspect

        tool = creation_result.unwrap()
        print(f"✓ Created tool: {tool.__name__}{inspect.signature(tool)}")
        print(f"docstring: {inspect.getdoc(tool)}")
    else:
        print(f"✗ Failed to create function: {creation_result.unwrap_err()}")
