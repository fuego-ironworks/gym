#!/usr/bin/env python3
"""Validate a Gym JSON record against the small schema subset used in this repo."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "enum",
    "minLength",
    "pattern",
    "minimum",
    "minItems",
    "items",
}


TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def schema_errors(schema: dict[str, Any], path: str = "$schema") -> list[str]:
    errors: list[str] = []
    unsupported = sorted(set(schema) - SUPPORTED_SCHEMA_KEYS)
    if unsupported:
        errors.append(f"{path}: unsupported schema keyword(s): {', '.join(unsupported)}")

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child_schema in properties.items():
            if isinstance(child_schema, dict):
                errors.extend(schema_errors(child_schema, f"{path}.properties.{name}"))

    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        errors.extend(schema_errors(item_schema, f"{path}.items"))

    return errors


def validate(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    errors = schema_errors(schema)
    if errors:
        return errors

    expected_type = schema.get("type")

    if expected_type is not None:
        check = TYPE_CHECKS.get(expected_type)
        if check is None:
            errors.append(f"{path}: unsupported schema type {expected_type!r}")
            return errors
        if not check(value):
            errors.append(f"{path}: expected {expected_type}")
            return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            errors.append(f"{path}: string shorter than {min_length}")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            errors.append(f"{path}: does not match {pattern!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: must be >= {minimum}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path}: needs at least {min_items} item(s)")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                errors.extend(validate(item_schema, item, f"{path}[{index}]"))

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")

        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected property {name!r}")

        for name, child_schema in properties.items():
            if name in value:
                errors.extend(validate(child_schema, value[name], f"{path}.{name}"))

    return errors


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} SCHEMA.json RECORD.json", file=sys.stderr)
        return 2

    schema_path = Path(argv[1])
    record_path = Path(argv[2])

    try:
        schema = load_json(schema_path)
        record = load_json(record_path)
    except (OSError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2

    errors = validate(schema, record)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"PASS {record_path} matches {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
