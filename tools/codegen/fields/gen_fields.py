#!/usr/bin/env python3
# ===========================================================================
# nk field-registry code generator (M3a).
#
# Reads src/nk/model/fields.yaml (THE single source of truth) and emits FOUR
# headers into src/nk/model/generated/:
#   field_ids.hpp    — enum class FieldId : uint16_t { <name>..., Count }
#   views.hpp        — nuka::phi::ModelView + nuka::phi::DataView (the REAL
#                      definitions phi/backend.hpp only forward-declares):
#                      typed device-pointer members, one per field, split by
#                      owner (model-owned constant tables vs data-owned state).
#   arena_layout.hpp — constexpr per-field {arena, owner, dtype, elem,
#                      per-unit} table for the Arena/Model layout computation.
#   dlpack_table.hpp — per-field {FieldId, dtype code, ndim hint, diff flag}
#                      rows for the future c_abi DLPack bridge.
#
# DETERMINISM CONTRACT (gate 4): the field ORDER is exactly the yaml order
# (no sorting, no dict iteration over unordered structures); two runs produce
# byte-identical output. Wired into tools/codegen/regen.py.
# ===========================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIELDS_YAML = REPO_ROOT / "src" / "nk" / "model" / "fields.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src" / "nk" / "model" / "generated"

BANNER = (
    "// GENERATED — do not edit; regen via tools/codegen/regen.py\n"
    "// Source of truth: src/nk/model/fields.yaml\n"
)

# dtype -> (C++ pointer element type, DLPack-style dtype code, lanes).
# The math vector/transform element types come from the engine math headers
# (math::Vec3 / math::Quat / math::Transform) and the articulation spatial
# types (float[6] / float[36]); we model them as their underlying float/uint
# storage in the layout table while exposing typed pointers in the views.
DTYPE_INFO: dict[str, dict[str, Any]] = {
    # name : { cpp (View member ptr element), code (dlpack), lanes (scalars/elem) }
    "f32":       {"cpp": "float",            "code": "kF32", "lanes": 1},
    "u32":       {"cpp": "uint32_t",         "code": "kU32", "lanes": 1},
    "u64":       {"cpp": "uint64_t",         "code": "kU64", "lanes": 1},
    "u8":        {"cpp": "uint8_t",          "code": "kU8",  "lanes": 1},
    "vec3":      {"cpp": "::nuka::math::Vec3",      "code": "kF32", "lanes": 3},
    "quat":      {"cpp": "::nuka::math::Quat",      "code": "kF32", "lanes": 4},
    "transform": {"cpp": "::nuka::math::Transform", "code": "kF32", "lanes": 7},
    "spatial6":  {"cpp": "::nuka::nk::Spatial6",    "code": "kF32", "lanes": 6},
    "mat36":     {"cpp": "::nuka::nk::Mat36",       "code": "kF32", "lanes": 36},
}

VALID_PER = {
    "env", "dof", "link", "body", "contact_slot", "row_slot", "slot_dof",
    "row_dof",
    "particle", "dist_con", "bend_con", "vol_con", "shape_match_slot",
    "shape_match_member",
    "env_dof2", "scalar",
    # Multi-articulation co-residence (K Go2 in one env): per-articulation count
    # unit (= articulations_per_env) and per-articulation M-tile unit
    # (= articulations_per_env * max_dof^2). At articulations_per_env == 1 these
    # collapse element-for-element onto `env` / `env_dof2` (the K==1 D1 invariant).
    "articulation", "articulation_dof2",
    # Per-articulation flat-DOF tile unit (= articulations_per_env * max_dof); the
    # general contact pipeline's per-artic qdot_flat tile (S3). At
    # articulations_per_env == 1 it collapses element-for-element onto `dof` (the
    # K==1 D1 invariant).
    "articulation_dof",
    # Per cloth aerodynamic-drag element (one per surface triangle). 0 elements
    # for any non-cloth or drag-free cook -> a zero-byte segment.
    "aero_tri",
}
VALID_ARENA = {"persistent", "scratch", "tape"}
VALID_OWNER = {"model", "data"}
VALID_FLAGS = {"diff", "param", "readout"}


class GenError(Exception):
    pass


def _camel(name: str) -> str:
    """snake_case field name -> CamelCase FieldId enumerator / member doc."""
    return "".join(part.capitalize() for part in name.split("_"))


def load_fields(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "fields" not in doc:
        raise GenError(f"{path}: top-level mapping with a 'fields' list required")
    fields = doc["fields"]
    if not isinstance(fields, list) or not fields:
        raise GenError(f"{path}: 'fields' must be a non-empty list")

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, f in enumerate(fields):
        if not isinstance(f, dict) or "name" not in f:
            raise GenError(f"{path}: field #{i} is not a mapping with a name")
        name = str(f["name"])
        if name in seen:
            raise GenError(f"{path}: duplicate field name '{name}'")
        seen.add(name)

        dtype = str(f.get("dtype", ""))
        if dtype not in DTYPE_INFO:
            raise GenError(f"{path}: field '{name}' has bad dtype '{dtype}'")
        per = str(f.get("per", ""))
        if per not in VALID_PER:
            raise GenError(f"{path}: field '{name}' has bad per '{per}'")
        arena = str(f.get("arena", ""))
        if arena not in VALID_ARENA:
            raise GenError(f"{path}: field '{name}' has bad arena '{arena}'")
        owner = str(f.get("owner", "data"))
        if owner not in VALID_OWNER:
            raise GenError(f"{path}: field '{name}' has bad owner '{owner}'")
        flags = f.get("flags", []) or []
        if not isinstance(flags, list) or any(fl not in VALID_FLAGS for fl in flags):
            raise GenError(f"{path}: field '{name}' has bad flags {flags!r}")
        elem = int(f.get("elem", 1))
        if elem < 1:
            raise GenError(f"{path}: field '{name}' has bad elem {elem}")
        count = f.get("count", None)
        if per == "scalar" and count is None:
            raise GenError(f"{path}: field '{name}' is per:scalar but has no count")

        out.append({
            "name": name,
            "camel": _camel(name),
            "dtype": dtype,
            "per": per,
            "arena": arena,
            "owner": owner,
            "flags": list(flags),
            "elem": elem,
            "count": None if count is None else str(count),
        })
    return out


def _doc_comment(f: dict[str, Any]) -> str:
    bits = [f"per:{f['per']}", f"arena:{f['arena']}", f"owner:{f['owner']}"]
    if f["elem"] != 1:
        bits.append(f"elem:{f['elem']}")
    if f["count"]:
        bits.append(f"count:{f['count']}")
    if f["flags"]:
        bits.append("flags:[" + ",".join(f["flags"]) + "]")
    return " ".join(bits)


def gen_field_ids(fields: list[dict[str, Any]]) -> str:
    lines = [BANNER, "#pragma once", "", "#include <cstdint>", "",
             "namespace nuka::nk {", "",
             "// One stable enumerator per field (yaml order). FieldId is the key into",
             "// the arena segment table and the View member binding.",
             "enum class FieldId : uint16_t {"]
    for f in fields:
        lines.append(f"    {f['camel']},  // {f['name']} ({_doc_comment(f)})")
    lines.append("    Count")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr int kFieldCount = static_cast<int>(FieldId::Count);")
    lines.append("")
    lines.append("} // namespace nuka::nk")
    lines.append("")
    return "\n".join(lines)


def gen_views(fields: list[dict[str, Any]]) -> str:
    model = [f for f in fields if f["owner"] == "model"]
    data = [f for f in fields if f["owner"] == "data"]

    def members(group: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for f in group:
            cpp = DTYPE_INFO[f["dtype"]]["cpp"]
            out.append(f"    {cpp}* {f['name']} = nullptr;  // {_doc_comment(f)}")
        return out

    lines = [
        BANNER, "#pragma once", "",
        "// The REAL definitions of nuka::phi::ModelView / nuka::phi::DataView that",
        "// phi/backend.hpp only FORWARD-DECLARES. Ops receive these by const-ref and",
        "// read typed device pointers, one per field. Clean owner split: ModelView",
        "// carries the cook-constant model tables; DataView carries the mutable per-",
        "// World state. Both are plain aggregates of raw device pointers (filled by",
        "// nk::Model::UploadTo and nk::Data respectively).",
        "",
        "#include <cstdint>",
        "",
        '#include "math/vec3.hpp"',
        '#include "math/quat.hpp"',
        '#include "math/transform.hpp"',
        "",
        "namespace nuka::nk {",
        "// Spatial / matrix element types for the articulation device state",
        "// (float[6] spatial vectors, float[36] 6x6 spatial matrices). Trivial",
        "// aggregates so a typed pointer indexes one element per link/row.",
        "struct Spatial6 { float v[6]; };",
        "struct Mat36    { float m[36]; };",
        "} // namespace nuka::nk",
        "",
        "namespace nuka::phi {",
        "",
        "// Model-owned, cook-constant tables. Pointers index into the ONE Model",
        "// device buffer (nk::Model::UploadTo packs + 256B-aligns the sections).",
        "struct ModelView {",
    ]
    lines += members(model)
    lines.append("};")
    lines.append("")
    lines.append("// Data-owned, mutable per-World state. Pointers index into the nk::Arena")
    lines.append("// (Persistent / Scratch / Tape phi Buffers), env-major.")
    lines.append("struct DataView {")
    lines += members(data)
    lines.append("};")
    lines.append("")
    lines.append("} // namespace nuka::phi")
    lines.append("")
    return "\n".join(lines)


def gen_arena_layout(fields: list[dict[str, Any]]) -> str:
    lines = [
        BANNER, "#pragma once", "",
        "// Per-field layout descriptor table, indexed by FieldId. The Arena and",
        "// Model use this to compute deterministic 256B-aligned segment offsets",
        "// (per-unit multiplier x env_count x capacity x elem_size). owner picks",
        "// the buffer (Model device buffer vs Arena Persistent/Scratch/Tape).",
        "",
        "#include <cstdint>",
        "",
        '#include "nk/model/generated/field_ids.hpp"',
        "",
        "namespace nuka::nk {",
        "",
        "enum class FieldArena : uint8_t { Persistent, Scratch, Tape };",
        "enum class FieldOwner : uint8_t { Model, Data };",
        "// The count-unit a field is sized by. ArenaLayout/Model resolve each to a",
        "// concrete element count from the Model capacities x env_count.",
        "enum class FieldPer : uint8_t {",
        "    Env, Dof, Link, Body, ContactSlot, RowSlot, SlotDof, RowDof, Particle,",
        "    DistCon, BendCon, VolCon, ShapeMatchSlot, ShapeMatchMember, EnvDof2,",
        "    Scalar,",
        "    // Multi-articulation co-residence: per-articulation (= "
        "articulations_per_env)",
        "    // and per-articulation M-tile (= articulations_per_env * max_dof^2).",
        "    // APPENDED so the existing enumerator values never shift.",
        "    Articulation, ArticulationDof2,",
        "    // Per-articulation flat-DOF tile (= articulations_per_env * max_dof);",
        "    // the general-contact per-artic qdot_flat tile (S3). APPENDED LAST.",
        "    ArticulationDof,",
        "    // Per cloth aerodynamic-drag element (= aero_tris_per_env). APPENDED LAST.",
        "    AeroTri",
        "};",
        "",
        "struct FieldLayout {",
        "    FieldArena arena;",
        "    FieldOwner owner;",
        "    FieldPer   per;",
        "    uint16_t   elem;       // packed elements per unit (e.g. rows=16)",
        "    uint16_t   lanes;      // scalar lanes per element (vec3=3, mat36=36)",
        "    uint16_t   elem_size;  // bytes per element (elem x lanes x sizeof scalar)",
        "    uint32_t   scalar_count;  // for FieldPer::Scalar (0 otherwise); see note",
        "};",
        "",
        "// NOTE: per:scalar fields with a symbolic count (e.g. mat_buckets =",
        "// num_buckets*8) carry scalar_count = 0 here; the Model resolves the real",
        "// count from its bucket table at layout time (the symbolic expression is",
        "// documented in fields.yaml and field_ids.hpp).",
        "",
    ]
    per_enum = {
        "env": "Env", "dof": "Dof", "link": "Link", "body": "Body",
        "contact_slot": "ContactSlot", "row_slot": "RowSlot", "slot_dof": "SlotDof",
        "row_dof": "RowDof",
        "particle": "Particle",
        "dist_con": "DistCon", "bend_con": "BendCon", "vol_con": "VolCon",
        "shape_match_slot": "ShapeMatchSlot", "shape_match_member": "ShapeMatchMember",
        "env_dof2": "EnvDof2", "scalar": "Scalar",
        "articulation": "Articulation", "articulation_dof2": "ArticulationDof2",
        "articulation_dof": "ArticulationDof",
        "aero_tri": "AeroTri",
    }
    scalar_size = {"f32": 4, "u32": 4, "u64": 8, "u8": 1, "vec3": 4, "quat": 4,
                   "transform": 4, "spatial6": 4, "mat36": 4}
    lines.append("inline constexpr FieldLayout kFieldLayout[kFieldCount] = {")
    for f in fields:
        info = DTYPE_INFO[f["dtype"]]
        lanes = info["lanes"]
        elem = f["elem"]
        esize = elem * lanes * scalar_size[f["dtype"]]
        arena = {"persistent": "Persistent", "scratch": "Scratch", "tape": "Tape"}[f["arena"]]
        owner = {"model": "Model", "data": "Data"}[f["owner"]]
        per = per_enum[f["per"]]
        lines.append(
            f"    {{FieldArena::{arena}, FieldOwner::{owner}, FieldPer::{per}, "
            f"{elem}, {lanes}, {esize}, 0}},  // {f['name']}"
        )
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr const FieldLayout& LayoutOf(FieldId id) {")
    lines.append("    return kFieldLayout[static_cast<int>(id)];")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace nuka::nk")
    lines.append("")
    return "\n".join(lines)


def gen_dlpack_table(fields: list[dict[str, Any]]) -> str:
    lines = [
        BANNER, "#pragma once", "",
        "// Per-field DLPack descriptor rows for the future c_abi buffer bridge",
        "// (plan §3.3: c_abi/buffer.cpp queries this table instead of hand-written",
        "// per-field branches). dtype_code is the scalar element code; ndim_hint is",
        "// a coarse rank hint (1 = flat scalar array, 2 = packed/vector array); diff",
        "// marks the tape-visible / autograd-exposed fields.",
        "",
        "#include <cstdint>",
        "",
        '#include "nk/model/generated/field_ids.hpp"',
        "",
        "namespace nuka::nk {",
        "",
        "enum class DlpackDtype : uint8_t { kF32, kU32, kU64, kU8 };",
        "",
        "struct DlpackRow {",
        "    FieldId     field;",
        "    DlpackDtype dtype_code;",
        "    uint8_t     ndim_hint;",
        "    bool        diff;",
        "    bool        readout;",
        "};",
        "",
        "inline constexpr DlpackRow kDlpackTable[kFieldCount] = {",
    ]
    for f in fields:
        code = DTYPE_INFO[f["dtype"]]["code"]
        lanes = DTYPE_INFO[f["dtype"]]["lanes"]
        ndim = 1 if (lanes == 1 and f["elem"] == 1) else 2
        diff = "true" if "diff" in f["flags"] else "false"
        readout = "true" if "readout" in f["flags"] else "false"
        lines.append(
            f"    {{FieldId::{f['camel']}, DlpackDtype::{code}, {ndim}, {diff}, {readout}}},"
            f"  // {f['name']}"
        )
    lines.append("};")
    lines.append("")
    lines.append("} // namespace nuka::nk")
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def generate(fields_yaml: Path, output_dir: Path) -> list[Path]:
    fields = load_fields(fields_yaml)
    outputs = {
        "field_ids.hpp": gen_field_ids(fields),
        "views.hpp": gen_views(fields),
        "arena_layout.hpp": gen_arena_layout(fields),
        "dlpack_table.hpp": gen_dlpack_table(fields),
    }
    written: list[Path] = []
    for name, content in outputs.items():
        path = output_dir / name
        _write(path, content)
        written.append(path)
    return written


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", type=Path, default=DEFAULT_FIELDS_YAML)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        outputs = generate(args.fields, args.output_dir)
    except GenError as exc:
        print(exc, file=sys.stderr)
        return 1
    for o in outputs:
        try:
            print(o.resolve().relative_to(REPO_ROOT).as_posix())
        except ValueError:
            print(o.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
