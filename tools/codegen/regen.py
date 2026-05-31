#!/usr/bin/env python3
"""Regenerate v0.1 Universal Row CUDA stubs from Row IR YAML."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path
from typing import Any

import jinja2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_DIR = REPO_ROOT / "tools" / "codegen"
DEFAULT_SCHEMA = CODEGEN_DIR / "schema" / "row_v0_1.yaml"
DEFAULT_CLASSES_DIR = CODEGEN_DIR / "classes"
DEFAULT_TEMPLATES_DIR = CODEGEN_DIR / "templates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src" / "codegen" / "generated"

GENERATED_HEADER = "// GENERATED — DO NOT EDIT"
ALLOWED_ROW_CLASSES = {
    "MaximalContactRow",
    "MaximalJointRow",
    "MaximalDriveRow",
    "FeatherstoneContactRow",
}


class CodegenError(Exception):
    """Schema or generation failure with a compiler-style diagnostic."""


@dataclasses.dataclass(frozen=True)
class SourceMap:
    path: Path
    key_lines: dict[str, int]

    def line_for(self, key: str) -> int:
        return self.key_lines.get(key, 1)


@dataclasses.dataclass(frozen=True)
class RowClass:
    data: dict[str, Any]
    source_path: Path
    source_map: SourceMap

    @property
    def row_class_id(self) -> int:
        return int(self.data["row_class_id"])

    @property
    def row_class_name(self) -> str:
        return str(self.data["row_class_name"])

    @property
    def file_stem(self) -> str:
        return snake_case(self.row_class_name.removesuffix("Row"))

    @property
    def has_adjoint(self) -> bool:
        return "adjoint_evaluator" in self.data

    def template_context(self) -> dict[str, Any]:
        context = dict(self.data)
        context["class_name"] = self.row_class_name.removesuffix("Row")
        context["row_class_name"] = self.row_class_name
        context["source_yaml_path"] = display_path(self.source_path)
        context["has_adjoint"] = self.has_adjoint
        return context


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def snake_case(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    return second.lower()


def field_cuda_type(field_name: str) -> str:
    if field_name in {"body_a", "body_b", "maximal_body", "articulation_id", "link_index"}:
        return "uint32_t"
    return "float"


def _diagnostic(path: Path, line: int, message: str) -> CodegenError:
    return CodegenError(f"{display_path(path)}:{line}: error: {message}")


def _load_yaml(path: Path) -> tuple[dict[str, Any], SourceMap]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _diagnostic(path, 1, str(exc)) from exc

    try:
        data = yaml.safe_load(text)
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = int(mark.line) + 1 if mark is not None else 1
        raise _diagnostic(path, line, str(exc).splitlines()[0]) from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise _diagnostic(path, 1, "YAML root must be a mapping")

    key_lines: dict[str, int] = {}
    if isinstance(root, yaml.MappingNode):
        for key_node, _ in root.value:
            if isinstance(key_node.value, str):
                key_lines[key_node.value] = int(key_node.start_mark.line) + 1

    return data, SourceMap(path=path, key_lines=key_lines)


def _require_mapping(value: Any, path: Path, line: int, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _diagnostic(path, line, f"field '{field}' must be a mapping")
    return value


def _require_list(value: Any, path: Path, line: int, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _diagnostic(path, line, f"field '{field}' must be a list")
    return value


def _validate_type(
    row: dict[str, Any],
    source_map: SourceMap,
    field: str,
    expected: str,
    enums: dict[str, list[str]],
) -> None:
    path = source_map.path
    value = row[field]
    line = source_map.line_for(field)

    if expected == "uint32":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _diagnostic(path, line, f"field '{field}' must be a non-negative integer")
        return
    if expected == "nullable_uint32":
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise _diagnostic(path, line, f"field '{field}' must be null or a non-negative integer")
        return
    if expected == "string":
        if not isinstance(value, str) or not value:
            raise _diagnostic(path, line, f"field '{field}' must be a non-empty string")
        return
    if expected == "bool":
        if not isinstance(value, bool):
            raise _diagnostic(path, line, f"field '{field}' must be a boolean")
        return
    if expected == "object":
        _require_mapping(value, path, line, field)
        return
    if expected == "enum":
        allowed = enums.get(field)
        if allowed is None:
            raise _diagnostic(path, line, f"schema missing enum declaration for '{field}'")
        if value not in allowed:
            allowed_text = ", ".join(allowed)
            raise _diagnostic(path, line, f"field '{field}' must be one of: {allowed_text}")
        return

    raise _diagnostic(path, line, f"unsupported schema type '{expected}' for field '{field}'")


def _validate_forward_evaluator(row: dict[str, Any], source_map: SourceMap, schema: dict[str, Any]) -> None:
    evaluator = _require_mapping(
        row["forward_evaluator"],
        source_map.path,
        source_map.line_for("forward_evaluator"),
        "forward_evaluator",
    )
    required = schema.get("forward_evaluator", {}).get("required_fields", [])
    if not isinstance(required, list):
        raise _diagnostic(source_map.path, 1, "schema forward_evaluator.required_fields must be a list")

    for field in required:
        if field not in evaluator:
            raise _diagnostic(
                source_map.path,
                source_map.line_for("forward_evaluator"),
                f"forward_evaluator missing required field '{field}'",
            )

    for field in ("input_fields", "output_fields"):
        values = _require_list(
            evaluator[field],
            source_map.path,
            source_map.line_for("forward_evaluator"),
            f"forward_evaluator.{field}",
        )
        if not values or not all(isinstance(item, str) and item for item in values):
            raise _diagnostic(
                source_map.path,
                source_map.line_for("forward_evaluator"),
                f"forward_evaluator.{field} must contain non-empty strings",
            )

    for field in ("body_index_field", "jacobian_field", "inner_loop_strategy"):
        if not isinstance(evaluator[field], str) or not evaluator[field]:
            raise _diagnostic(
                source_map.path,
                source_map.line_for("forward_evaluator"),
                f"forward_evaluator.{field} must be a non-empty string",
            )

    if not isinstance(evaluator["warm_start"], bool):
        raise _diagnostic(
            source_map.path,
            source_map.line_for("forward_evaluator"),
            "forward_evaluator.warm_start must be a boolean",
        )


def _validate_adjoint_evaluator(row: dict[str, Any], source_map: SourceMap, schema: dict[str, Any]) -> None:
    """Validate the OPTIONAL adjoint_evaluator block (only if present)."""
    if "adjoint_evaluator" not in row:
        return

    line = source_map.line_for("adjoint_evaluator")
    evaluator = _require_mapping(row["adjoint_evaluator"], source_map.path, line, "adjoint_evaluator")
    required = schema.get("adjoint_evaluator", {}).get("required_fields", [])
    if not isinstance(required, list):
        raise _diagnostic(source_map.path, 1, "schema adjoint_evaluator.required_fields must be a list")

    for field in required:
        if field not in evaluator:
            raise _diagnostic(source_map.path, line, f"adjoint_evaluator missing required field '{field}'")

    grad_inputs = _require_list(evaluator["grad_input_fields"], source_map.path, line, "adjoint_evaluator.grad_input_fields")
    if not grad_inputs or not all(isinstance(item, str) and item for item in grad_inputs):
        raise _diagnostic(source_map.path, line, "adjoint_evaluator.grad_input_fields must contain non-empty strings")

    grad_outputs = _require_list(evaluator["grad_output_fields"], source_map.path, line, "adjoint_evaluator.grad_output_fields")
    if len(grad_outputs) != 1 or not all(isinstance(item, str) and item for item in grad_outputs):
        raise _diagnostic(source_map.path, line, "adjoint_evaluator.grad_output_fields must be a single non-empty string (v0.5 supports one scalar output)")

    if not isinstance(evaluator["unclamped_expression"], str) or not evaluator["unclamped_expression"].strip():
        raise _diagnostic(source_map.path, line, "adjoint_evaluator.unclamped_expression must be a non-empty string")

    rules = _require_mapping(evaluator["derivative_rules"], source_map.path, line, "adjoint_evaluator.derivative_rules")
    for field in grad_inputs:
        if field not in rules:
            raise _diagnostic(source_map.path, line, f"adjoint_evaluator.derivative_rules missing rule for grad input '{field}'")
        if not isinstance(rules[field], str) or not rules[field].strip():
            raise _diagnostic(source_map.path, line, f"adjoint_evaluator.derivative_rules['{field}'] must be a non-empty C expression")


def _validate_row(row: dict[str, Any], source_map: SourceMap, schema: dict[str, Any]) -> RowClass:
    required = schema.get("required_fields", [])
    field_types = schema.get("field_types", {})
    enums = schema.get("enums", {})
    if not isinstance(required, list) or not isinstance(field_types, dict) or not isinstance(enums, dict):
        raise _diagnostic(source_map.path, 1, "schema must define required_fields, field_types, and enums")

    for field in required:
        if field not in row:
            raise _diagnostic(source_map.path, 1, f"missing required field '{field}'")

    for field in required:
        expected = field_types.get(field)
        if expected is None:
            raise _diagnostic(source_map.path, source_map.line_for(field), f"schema missing type for '{field}'")
        _validate_type(row, source_map, field, expected, enums)

    row_name = row["row_class_name"]
    if row_name not in ALLOWED_ROW_CLASSES:
        allowed = ", ".join(sorted(ALLOWED_ROW_CLASSES))
        raise _diagnostic(source_map.path, source_map.line_for("row_class_name"), f"unsupported v0.1 row class '{row_name}'; allowed: {allowed}")
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*Row", row_name):
        raise _diagnostic(source_map.path, source_map.line_for("row_class_name"), "row_class_name must be CamelCase and end with Row")

    mode = row["body_count_mode"]
    body_count = row["body_count"]
    if mode == "fixed" and body_count is None:
        raise _diagnostic(source_map.path, source_map.line_for("body_count"), "fixed body_count_mode requires body_count")
    if mode == "variable" and body_count is not None:
        raise _diagnostic(source_map.path, source_map.line_for("body_count"), "variable body_count_mode requires body_count: null")
    if isinstance(body_count, int) and body_count == 0:
        raise _diagnostic(source_map.path, source_map.line_for("body_count"), "body_count must be greater than zero")
    if row["max_rows_per_block"] == 0:
        raise _diagnostic(source_map.path, source_map.line_for("max_rows_per_block"), "max_rows_per_block must be greater than zero")

    _validate_forward_evaluator(row, source_map, schema)
    _validate_adjoint_evaluator(row, source_map, schema)
    return RowClass(data=row, source_path=source_map.path, source_map=source_map)


def load_rows(schema_path: Path, classes_dir: Path) -> tuple[dict[str, Any], list[RowClass]]:
    schema, _ = _load_yaml(schema_path)
    class_paths = sorted(classes_dir.glob("*.yaml"))
    if not class_paths:
        raise _diagnostic(classes_dir, 1, "no row class YAML files found")

    rows: list[RowClass] = []
    ids: dict[int, Path] = {}
    names: dict[str, Path] = {}
    for path in class_paths:
        row_data, source_map = _load_yaml(path)
        row = _validate_row(row_data, source_map, schema)
        if row.row_class_id in ids:
            raise _diagnostic(path, source_map.line_for("row_class_id"), f"duplicate row_class_id {row.row_class_id}; first seen in {display_path(ids[row.row_class_id])}")
        if row.row_class_name in names:
            raise _diagnostic(path, source_map.line_for("row_class_name"), f"duplicate row_class_name {row.row_class_name}; first seen in {display_path(names[row.row_class_name])}")
        ids[row.row_class_id] = path
        names[row.row_class_name] = path
        rows.append(row)

    missing = ALLOWED_ROW_CLASSES.difference(names)
    if missing:
        raise _diagnostic(classes_dir, 1, f"missing v0.1 row class IR files: {', '.join(sorted(missing))}")

    return schema, sorted(rows, key=lambda item: item.row_class_id)


def _environment(templates_dir: Path) -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["snake_case"] = snake_case
    env.globals["field_cuda_type"] = field_cuda_type
    return env


def _write(path: Path, content: str) -> None:
    if not content.startswith(GENERATED_HEADER):
        raise _diagnostic(path, 1, f"generated output must start with '{GENERATED_HEADER}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _gradient_mode_enum(schema: dict[str, Any]) -> list[str]:
    """Ordered gradient-mode names; index == the uint8 stored in Row.gradient_mode."""
    modes = schema.get("enums", {}).get("default_gradient_mode", [])
    if not isinstance(modes, list) or not modes:
        raise CodegenError("schema enums.default_gradient_mode must be a non-empty list")
    return [str(mode) for mode in modes]


def render(schema: dict[str, Any], rows: list[RowClass], templates_dir: Path, output_dir: Path) -> list[Path]:
    env = _environment(templates_dir)
    forward_template = env.get_template("forward_kernel.cu.j2")
    dispatch_template = env.get_template("row_dispatch.cu.j2")
    registry_template = env.get_template("registry_header.hpp.j2")
    adjoint_template = env.get_template("adjoint_kernel.cuh.j2")
    adjoint_kernel_template = env.get_template("adjoint_kernel.cu.j2")

    gradient_modes = _gradient_mode_enum(schema)

    # Assign a stable, non-zero adjoint_kernel_id to each class that emits an
    # adjoint (id 0 == "no adjoint"); classes without one keep 0.
    row_contexts: list[dict[str, Any]] = []
    next_adjoint_id = 1
    for row in rows:
        ctx = row.template_context()
        mode = str(row.data["default_gradient_mode"])
        ctx["gradient_mode_index"] = gradient_modes.index(mode)
        if row.has_adjoint:
            ctx["adjoint_kernel_id"] = next_adjoint_id
            next_adjoint_id += 1
        else:
            ctx["adjoint_kernel_id"] = 0
        row_contexts.append(ctx)

    common = {
        "rows": row_contexts,
        "flags": schema.get("flags_bitfield", []),
        "gradient_modes": gradient_modes,
    }

    outputs: list[Path] = []
    for row, ctx in zip(rows, row_contexts):
        output = output_dir / f"{row.file_stem}_forward.cu"
        _write(output, forward_template.render(**common, **ctx))
        outputs.append(output)

        # Reverse-mode pair: emit only for classes carrying an adjoint_evaluator.
        if row.has_adjoint:
            header = output_dir / f"{row.file_stem}_adjoint.cuh"
            _write(header, adjoint_template.render(**common, **ctx))
            outputs.append(header)

            kernel = output_dir / f"{row.file_stem}_adjoint.cu"
            _write(kernel, adjoint_kernel_template.render(**common, **ctx))
            outputs.append(kernel)

    dispatch = output_dir / "row_dispatch.cu"
    _write(dispatch, dispatch_template.render(**common))
    outputs.append(dispatch)

    registry = output_dir / "row_class_registry.hpp"
    _write(registry, registry_template.render(**common))
    outputs.append(registry)

    return outputs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--classes-dir", type=Path, default=DEFAULT_CLASSES_DIR)
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        schema, rows = load_rows(args.schema, args.classes_dir)
        outputs = render(schema, rows, args.templates_dir, args.output_dir)
    except (CodegenError, jinja2.TemplateError) as exc:
        print(exc, file=sys.stderr)
        return 1

    for output in outputs:
        print(display_path(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
