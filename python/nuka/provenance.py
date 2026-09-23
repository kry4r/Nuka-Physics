"""Preserve the asset, source and loaded engine identity of a local run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def record_run_identity(root: Path, output: Path, inputs, source_paths):
    root, output = Path(root).resolve(), Path(output).resolve()
    assets = {}

    def visit(path):
        path = Path(path).resolve()
        if str(path) in assets:
            return
        assets[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix != ".nks":
            return

        def references(value):
            if isinstance(value, dict):
                for child in value.values():
                    references(child)
            elif isinstance(value, list):
                for child in value:
                    references(child)
            elif isinstance(value, str):
                candidate = path.parent / value.split("#", 1)[0]
                if candidate.is_file():
                    visit(candidate)

        references(json.loads(path.read_text(encoding="utf-8")))

    for path in inputs:
        if path is not None:
            visit(path)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root)
    patch = git("diff", "--binary", "HEAD", "--", *source_paths)
    (output / "source.patch").write_bytes(patch)
    untracked = {}
    names = git("ls-files", "--others", "--exclude-standard", "--", *source_paths).decode().splitlines()
    for name in names:
        source = root / name
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        untracked[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    libraries = {}
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) < 6:
                continue
            path = Path(parts[5])
            if not path.name.startswith(("libnuka", "_nuka_ext")) or str(path) in libraries:
                continue
            if not path.is_file():
                raise RuntimeError(f"The loaded engine library changed during the run: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archive = root / ".nuka-runs/artifacts/binaries" / digest / path.name
            if not archive.exists():
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, archive)
            libraries[str(path)] = {"sha256": digest, "archive": str(archive)}
    identity = {"head": git("rev-parse", "HEAD").decode().strip(), "inputs": assets,
                "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
                "untracked_source": untracked, "libraries": libraries,
                "python": platform.python_version(), "platform": platform.platform(),
                "command": sys.argv, "library_path": os.environ.get("LD_LIBRARY_PATH", "")}
    (output / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity
