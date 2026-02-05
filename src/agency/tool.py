import inspect
from typing import get_type_hints


def _parse_parameter_descriptions(docstring):
    parameter_descriptions = {}
    in_args_section = False

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        if not in_args_section:
            continue

        if not stripped or (not line[0].isspace() if line else True):
            break

        if ":" not in stripped:
            continue

        name, description = stripped.split(":", 1)
        parameter_descriptions[name.strip()] = description.strip()

    return parameter_descriptions


def tool(function=None, *, name=None, description=None):
    def decorator(function):
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

        tool_name = name or function.__name__
        hints = get_type_hints(function)
        signature = inspect.signature(function)
        docstring = inspect.getdoc(function) or ""

        tool_description = description or (
            docstring.strip().split("\n")[0] if docstring.strip() else function.__name__
        )
        parameter_descriptions = _parse_parameter_descriptions(docstring)

        properties = {}
        required = []

        for parameter_name, parameter in signature.parameters.items():
            property_schema = {"type": type_map.get(hints.get(parameter_name), "string")}

            if parameter_name in parameter_descriptions:
                property_schema["description"] = parameter_descriptions[parameter_name]

            properties[parameter_name] = property_schema

            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)

        function.schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return function

    if function is not None:
        return decorator(function)

    return decorator
