#!/usr/bin/env python3
"""Physics-smell lint for CUDA-first simulation guardrails."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import time
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by local environment.
    raise SystemExit("physics_smell.py requires PyYAML; install pyyaml") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_DIR = Path(__file__).resolve().parent
PATTERNS_PATH = LINT_DIR / "banned_patterns.yaml"
ALLOWLIST_PATH = LINT_DIR / "allowlist.yaml"


@dataclasses.dataclass(frozen=True)
class Pattern:
    id: str
    description: str
    regex: re.Pattern[str]
    scope: str
    severity: str


@dataclasses.dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    pattern_id: str
    description: str
    matched_text: str


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def _matches_glob(path: Path, patterns: Iterable[str]) -> bool:
    normalized = path.as_posix()
    return any(path.match(pattern) or re.fullmatch(_glob_to_regex(pattern), normalized) for pattern in patterns)


def _glob_to_regex(pattern: str) -> str:
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*\*/", r"(?:.*/)?")
    escaped = escaped.replace(r"\*\*", r".*")
    escaped = escaped.replace(r"\*", r"[^/]*")
    return escaped


def _scope_files(scope: dict, selected_files: set[Path] | None) -> list[Path]:
    includes = scope.get("include", [])
    excludes = scope.get("exclude", [])
    files: set[Path] = set()
    for pattern in includes:
        files.update(path for path in REPO_ROOT.glob(pattern) if path.is_file())

    rel_files = {_relative(path) for path in files}
    rel_files = {path for path in rel_files if not _matches_glob(path, excludes)}
    if selected_files is not None:
        rel_files = {path for path in rel_files if path in selected_files}
    return sorted(rel_files)


def _load_patterns(selected_pattern: str | None) -> tuple[list[Pattern], dict]:
    config = _load_yaml(PATTERNS_PATH)
    raw_patterns = config.get("patterns", [])
    patterns: list[Pattern] = []
    for raw in raw_patterns:
        if selected_pattern is not None and raw["id"] != selected_pattern:
            continue
        patterns.append(
            Pattern(
                id=raw["id"],
                description=raw["description"],
                regex=re.compile(raw["regex"]),
                scope=raw["scope"],
                severity=raw.get("severity", "error"),
            )
        )
    if selected_pattern is not None and not patterns:
        known = ", ".join(item["id"] for item in raw_patterns)
        raise SystemExit(f"unknown pattern '{selected_pattern}'. Known patterns: {known}")
    return patterns, config.get("scope_definitions", {})


def _load_allowlist() -> list[dict]:
    entries = _load_yaml(ALLOWLIST_PATH).get("allowlist", [])
    for entry in entries:
        missing = [key for key in ("path", "pattern", "reason") if not entry.get(key)]
        if missing:
            raise SystemExit(f"allowlist entry missing required keys {missing}: {entry}")
        if not entry.get("line") and not entry.get("match"):
            raise SystemExit(f"allowlist entry must specify line or match: {entry}")
    return entries


def _allowed(violation: Violation, allowlist: list[dict]) -> bool:
    rel = violation.path.as_posix()
    for entry in allowlist:
        if entry["path"] != rel or entry["pattern"] != violation.pattern_id:
            continue
        line_ok = not entry.get("line") or int(entry["line"]) == violation.line
        match_ok = not entry.get("match") or entry["match"] in violation.matched_text
        if line_ok and match_ok:
            return True
    return False


def _scan_file(path: Path, pattern: Pattern) -> list[Violation]:
    full_path = REPO_ROOT / path
    violations: list[Violation] = []
    try:
        lines = full_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_no, line in enumerate(lines, start=1):
        match = pattern.regex.search(line)
        if match:
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    pattern_id=pattern.id,
                    description=pattern.description,
                    matched_text=match.group(0),
                )
            )
    return violations


def scan(selected_pattern: str | None, selected_files: list[str] | None) -> list[Violation]:
    patterns, scopes = _load_patterns(selected_pattern)
    selected_paths = None
    if selected_files is not None:
        selected_paths = {_relative((REPO_ROOT / item).resolve()) for item in selected_files}

    allowlist = _load_allowlist()
    violations: list[Violation] = []
    for pattern in patterns:
        if pattern.scope not in scopes:
            raise SystemExit(f"pattern {pattern.id} references missing scope {pattern.scope}")
        for path in _scope_files(scopes[pattern.scope], selected_paths):
            violations.extend(_scan_file(path, pattern))

    return [violation for violation in violations if not _allowed(violation, allowlist)]


def _print_violations(violations: list[Violation]) -> None:
    for violation in violations:
        print(
            f"{violation.path}:{violation.line}: error: "
            f"[{violation.pattern_id}] {violation.description}"
        )


def _fix_allowlist(selected_pattern: str | None, selected_files: list[str] | None) -> int:
    violations = scan(selected_pattern, selected_files)
    if not violations:
        print("No violations to allowlist.")
        return 0

    print("The following violations are not allowlisted:")
    _print_violations(violations)
    answer = input("Append these entries to tools/lint/allowlist.yaml? [y/N] ")
    if answer.lower() not in {"y", "yes"}:
        return 1

    with ALLOWLIST_PATH.open("a", encoding="utf-8") as handle:
        for violation in violations:
            handle.write(
                "\n"
                f"  - path: {violation.path.as_posix()}\n"
                f"    pattern: {violation.pattern_id}\n"
                f"    line: {violation.line}\n"
                f"    match: {violation.matched_text!r}\n"
                "    reason: \"TODO: explain why this exception is safe\"\n"
            )
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", help="scan only one banned pattern id")
    parser.add_argument("--files", nargs="+", help="scan only these repository-relative files")
    parser.add_argument(
        "--fix-allowlist",
        action="store_true",
        help="interactively append current violations to tools/lint/allowlist.yaml",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="print scan duration to stderr",
    )
    args = parser.parse_args(argv)

    if args.fix_allowlist:
        return _fix_allowlist(args.pattern, args.files)

    started = time.perf_counter()
    violations = scan(args.pattern, args.files)
    elapsed = time.perf_counter() - started
    _print_violations(violations)
    if args.time:
        print(f"physics_smell.py: scanned in {elapsed:.3f}s", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
