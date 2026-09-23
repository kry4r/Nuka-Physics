"""Fetch the jacket used by Newton's cloth_h1 example with its license."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


REVISION = "a0547548eaa966c2f5478bee496c3cfba1fa98fc"
FILES = {
    "style3d/garments/h1_jacket.usd": "h1_jacket.usd",
    "style3d/LICENSE": "LICENSE",
    "style3d/README.md": "README.md",
    "unitree_h1/mjcf/h1_with_hand.xml": "h1_with_hand.xml",
    "unitree_h1/LICENSE": "H1_LICENSE",
}


def fetch(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "https://github.com/newton-physics/newton-assets",
                "revision": REVISION, "files": {}}
    manifest_path = destination / "manifest.json"
    cached = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for source, name in FILES.items():
        url = f"https://raw.githubusercontent.com/newton-physics/newton-assets/{REVISION}/{source}"
        path = destination / name
        previous = cached.get("files", {}).get(name, {})
        data = path.read_bytes() if path.exists() else b""
        valid = (cached.get("revision") == REVISION and previous.get("url") == url
                 and hashlib.sha256(data).hexdigest() == previous.get("sha256"))
        if not valid:
            request = urllib.request.Request(url, headers={"User-Agent": "Nuka-Physics asset fetcher"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        if not data or data.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Missing asset payload: {source}")
        if not valid:
            path.write_bytes(data)
        manifest["files"][name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(),
                                    "bytes": len(data)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / ".nuka-assets/garments/newton_h1_jacket")
    args = parser.parse_args()
    print(json.dumps(fetch(args.out), indent=2))


if __name__ == "__main__":
    main()
