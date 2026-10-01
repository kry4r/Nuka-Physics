"""Fetch reproducible Unitree towel episode candidates and source metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


REPOSITORY = "unitreerobotics/G1_Dex1_Fold_Towel"
REVISION = "e004cbddc8ba94110773dc9155eb77273a2c2ef8"
HOST = "https://huggingface.co"


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "Nuka-Physics asset fetcher"})


def verify(path, record):
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {record['size']}\0".encode())
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            blob.update(chunk)
            size += len(chunk)
    if size != record["size"]:
        raise ValueError(f"Asset size mismatch: {path}")
    lfs = record.get("lfs")
    if lfs and digest.hexdigest() != lfs["sha256"]:
        raise ValueError(f"Asset SHA256 mismatch: {path}")
    if not lfs and blob.hexdigest() != record["blobId"]:
        raise ValueError(f"Asset Git blob mismatch: {path}")
    return {"sha256": digest.hexdigest(), "bytes": size, "source_blob": record["blobId"]}


def fetch(destination, episodes, cameras, metadata_only=False):
    url = f"{HOST}/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(request(url), timeout=60) as response:
        source = json.load(response)
    if source["sha"] != REVISION:
        raise ValueError("Dataset metadata does not match the pinned revision")
    files = {entry["rfilename"]: entry for entry in source["siblings"]}
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": f"{HOST}/datasets/{REPOSITORY}", "revision": REVISION,
                "candidate_episodes": episodes, "cameras": cameras,
                "selection_status": "candidates_awaiting_full_video_review", "files": {}}

    def download(name):
        if name not in files:
            raise ValueError(f"Missing file at pinned revision: {name}")
        path = destination / name
        asset_url = f"{HOST}/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".download-",
                                             delete=False) as output:
                partial = Path(output.name)
                with urllib.request.urlopen(request(asset_url), timeout=60) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            verify(partial, files[name])
            partial.replace(path)
        manifest["files"][name] = {"url": asset_url, **verify(path, files[name])}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"file": name, "bytes": manifest["files"][name]["bytes"]}), flush=True)
        return path

    for name in ("README.md", "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl",
                 "meta/episodes_stats.jsonl"):
        download(name)
    info = json.loads((destination / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["fps"] != 30:
        raise ValueError("Unsupported episode schema or frame rate")
    if any(index < 0 or index >= info["total_episodes"] for index in episodes):
        raise ValueError("Episode index outside the pinned dataset")
    for camera in cameras:
        if info["features"].get(camera, {}).get("dtype") != "video":
            raise ValueError(f"Unknown video camera: {camera}")
    if not metadata_only:
        for index in episodes:
            chunk = index // info["chunks_size"]
            values = {"episode_chunk": chunk, "episode_index": index}
            download(info["data_path"].format(**values))
            for camera in cameras:
                download(info["video_path"].format(video_key=camera, **values))
    manifest["total_episodes"] = info["total_episodes"]
    manifest["fps"] = info["fps"]
    manifest["metadata_only"] = metadata_only
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / ".nuka-assets/datasets/g1_fold_towel")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cameras", nargs="+", default=["observation.images.cam_left_high",
                                                       "observation.images.cam_right_high"])
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    result = fetch(args.out, list(dict.fromkeys(args.episodes)), list(dict.fromkeys(args.cameras)),
                   args.metadata_only)
    print(json.dumps({"revision": result["revision"], "files": len(result["files"]),
                      "selection_status": result["selection_status"]}))


if __name__ == "__main__":
    main()
