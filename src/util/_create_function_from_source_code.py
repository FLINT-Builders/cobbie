import textwrap
from typing import Any, Callable, Dict, Optional

from src.schemas import Result, ok, err
from src.util.sandbox import source_tool
from ._extract_function_metadata import _extract_function_metadata


def _create_function_from_source_code(
    code: str,
    function_name: str,
    imports: Optional[Dict[str, Any]] = None,
    merge_with_defaults: bool = True,
) -> Result[Callable, str]:
    """
    Create a callable tool from the generated code that can be used by the assessor.

    The code is parsed, never executed here: the returned tool describes the
    function (name, signature, docstring) and only runs inside the sandboxed
    interpreter, where it is defined with the default helper imports
    (ifcopenshell and its utilities, math, numpy, pandas). Exceptions raised by
    the tool are returned as an error string, as before.

    Args:
        code: The source code containing the function to create
        function_name: The name of the function to extract from the code
        imports: Not supported any more; generated tools get the default imports
                 inside the sandbox.
        merge_with_defaults: Kept for backward compatibility.

    Returns:
        Result[Callable, str]: Ok(function) on success, Err(error_message) on failure.
                              Use .is_ok() and .is_err() to check the result status.
    """
    if imports is not None:
        return err("Custom imports are not supported: generated tools run in the sandbox with the default imports")

    # Clean up the code string - remove common leading whitespace
    cleaned_code = textwrap.dedent(code).strip()

    try:
        # Extract function metadata to get the signature
        metadata = _extract_function_metadata(cleaned_code, function_name)
        tool_wrapper = source_tool(cleaned_code, function_name, wrap_errors=True)

        # Set the function name and docstring for the tool
        tool_wrapper.__name__ = function_name

        # Create a more descriptive docstring
        args_description = ""
        if metadata.args:
            args_list = []
            for i, arg in enumerate(metadata.args):
                # Check if this argument has a default value
                default_index = i - (len(metadata.args) - len(metadata.defaults))
                if default_index >= 0 and default_index < len(metadata.defaults):
                    default_val = metadata.defaults[default_index]
                    args_list.append(
                        f"            {arg}: Parameter with default value {default_val}"
                    )
                else:
                    args_list.append(f"            {arg}: Required parameter")
            args_description = "\n".join(args_list)

        # Create intelligent return type description
        return_description = "The actual return value of the function"
        if metadata.return_type:
            return_description = (
                f"{metadata.return_type}: As specified in the function signature"
            )
        elif metadata.docstring:
            # Try to extract return info from docstring more intelligently
            docstring_lines = metadata.docstring.split("\n")
            returns_found = False
            for i, line in enumerate(docstring_lines):
                line_stripped = line.strip()
                line_lower = line_stripped.lower()

                # Look for "Returns:" section
                if line_lower.startswith("returns:"):
                    returns_found = True
                    # Get the description from this line if it exists
                    if (
                        ":" in line_stripped
                        and len(line_stripped.split(":", 1)[1].strip()) > 0
                    ):
                        return_description = line_stripped.split(":", 1)[1].strip()
                        break
                    # Otherwise, look for description in next non-empty lines
                    elif i + 1 < len(docstring_lines):
                        for j in range(i + 1, len(docstring_lines)):
                            next_line = docstring_lines[j].strip()
                            if next_line and not next_line.lower().startswith(
                                ("args:", "parameters:", "raises:", "examples:")
                            ):
                                return_description = next_line
                                break
                    break

            # If no "Returns:" found, use a generic description
            if not returns_found:
                return_description = "Return type not specified."
        else:
            return_description = "Return type not specified."

        tool_wrapper.__doc__ = f"""
        {metadata.docstring or f"Tool: {function_name}"}

        Args:
            {args_description}

        Returns:
            {return_description}
        """

        return ok(tool_wrapper)

    except Exception as e:
        # Return error result instead of dummy function
        error_message = str(e)
        return err(error_message)


if __name__ == "__main__":
    # Import here to avoid circular imports in main execution
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    python_code = '''import ifcopenshell
import ifcopenshell.util.element
from typing import List

def get_emergency_exit_doors(path_ifc_model: str):
    """
    Retrieves a list of IfcDoor entities that are designated as emergency exits.

    Args:
    path_ifc_model (str): Path to the IFC file.

    Returns:
    List[ifcopenshell.entity_instance.entity_instance]: A list of IfcDoor entities representing emergency exits.
    """
    # Load the IFC file from the provided path
    ifc_file = ifcopenshell.open(path_ifc_model)

    # Retrieve all IfcDoor entities
    doors = ifc_file.by_type('IfcDoor')

    # Initialize an empty list to store emergency exit doors
    emergency_exit_doors = []

    # Iterate through each IfcDoor entity
    for door in doors:
        # Check if the door has a property set that indicates it's an emergency exit
        psets = ifcopenshell.util.element.get_psets(door)
        for pset_name, properties in psets.items():
            if 'EmergencyExit' in pset_name or 'emergencyexit' in (prop.lower() for prop in properties):
                emergency_exit_doors.append(door)
                break

    return emergency_exit_doors'''
    from src.config import TEST_IFC_PATH

    function_name = "get_emergency_exit_doors"
    creation_result = _create_function_from_source_code(
        function_name="get_emergency_exit_doors", code=python_code
    )

    if creation_result.is_ok():
        from src.util.python_executor import execute_python

        print(f"✓ Successfully created function: {function_name}")
        get_emergency_exit_doors = creation_result.unwrap()
        result = execute_python(
            f"print({function_name}(path_ifc_model))",
            tools={function_name: get_emergency_exit_doors},
            model_path=TEST_IFC_PATH,
        )
        print(f"---\nResult of function {function_name}:\n{result}\n")
        print(
            f"docstring of function: {function_name}:\n {get_emergency_exit_doors.__doc__}"
        )
        print("test completed")
    else:
        print(f"✗ Failed to create function: {creation_result.unwrap_err()}")
        print("test failed")
