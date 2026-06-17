"""Serve FSQ checkpoint rollout artifacts with a single static viewer app."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


VIEWER_DIR = Path(__file__).resolve().parent / "fsq_viewer"
SKIP_DIRS = {
    ".git",
    ".hydra",
    "__pycache__",
    "examples",
    "venv",
    "wandb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help=(
            "Root directory to scan for fsq_usage.json files. Can be passed "
            "multiple times."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def default_roots() -> list[Path]:
    candidates = [
        Path("outputs/fsq_diagnostics"),
        Path(os.environ.get("SCRATCH", "/scratch/tangzach")) / "jaxmarl",
    ]
    return [path.resolve() for path in candidates if path.exists()]


def iter_usage_paths(roots: list[Path]):
    seen = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname not in SKIP_DIRS and not dirname.startswith(".")
            ]
            if "fsq_usage.json" not in filenames:
                continue
            path = Path(dirpath) / "fsq_usage.json"
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def infer_run_name(viewer_dir: Path, metadata: dict) -> str:
    run_name = metadata.get("run_name")
    if run_name:
        return str(run_name)
    parts = viewer_dir.parts
    if "checkpoint_rollouts" in parts:
        idx = parts.index("checkpoint_rollouts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return viewer_dir.name


def infer_checkpoint(viewer_dir: Path, metadata: dict) -> int | str:
    update = metadata.get("checkpoint_update")
    if update is not None:
        return int(update)
    match = re.search(r"update_0*(\d+)", viewer_dir.name)
    if match:
        return int(match.group(1))
    return "unknown"


def infer_recipe(metadata: dict) -> tuple[int | None, str]:
    recipe_index = metadata.get("recipe_index")
    recipe = metadata.get("recipe") or "sampled"
    if recipe_index is None:
        return None, str(recipe)
    return int(recipe_index), str(recipe)


def load_artifacts(roots: list[Path]) -> list[dict]:
    artifacts = []
    for usage_path in iter_usage_paths(roots):
        try:
            data = json.loads(usage_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        viewer_dir = usage_path.parent
        metadata = data.get("metadata") or {}
        recipe_index, recipe = infer_recipe(metadata)
        checkpoint = infer_checkpoint(viewer_dir, metadata)
        artifacts.append(
            {
                "run": infer_run_name(viewer_dir, metadata),
                "checkpoint": checkpoint,
                "recipe_index": recipe_index,
                "recipe": recipe,
                "layout": data.get("layout"),
                "levels": data.get("levels"),
                "total_samples": data.get("total_samples", 0),
                "nonzero_codes": sum(
                    1 for code in data.get("codes", []) if code.get("count", 0)
                ),
                "path": str(viewer_dir),
                "metadata": metadata,
            }
        )

    artifacts = sorted(
        artifacts,
        key=lambda item: (
            item["run"],
            item["checkpoint"] if isinstance(item["checkpoint"], int) else -1,
            -1 if item["recipe_index"] is None else item["recipe_index"],
            item["recipe"],
        ),
    )
    for idx, artifact in enumerate(artifacts):
        artifact["id"] = f"a{idx}"
        artifact["usage_url"] = f"/artifact/a{idx}/fsq_usage.json"
        artifact["asset_url"] = f"/artifact/a{idx}/"
    return artifacts


class FSQViewerHandler(BaseHTTPRequestHandler):
    artifacts: list[dict] = []
    artifact_dirs: dict[str, Path] = {}
    allowed_roots: list[Path] = []
    viewer_dir: Path = VIEWER_DIR

    def do_GET(self):
        self._handle_request(send_body=True)

    def do_HEAD(self):
        self._handle_request(send_body=False)

    def _handle_request(self, *, send_body: bool):
        parsed = urlparse(self.path)
        if parsed.path == "/api/artifacts":
            payload = json.dumps({"artifacts": self.artifacts}).encode()
            self._send_bytes(payload, "application/json", send_body=send_body)
            return
        if parsed.path.startswith("/artifact/"):
            self._serve_artifact_path(parsed.path, send_body=send_body)
            return
        self._serve_viewer_path(parsed.path, send_body=send_body)

    def _send_bytes(self, payload: bytes, content_type: str, *, send_body: bool):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _serve_file(self, target: Path, *, send_body: bool):
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        if send_body:
            with target.open("rb") as file:
                self.wfile.write(file.read())

    def _serve_viewer_path(self, request_path: str, *, send_body: bool):
        if request_path in ("", "/"):
            rel = Path("index.html")
        else:
            rel = Path(unquote(request_path.lstrip("/")))
        if rel.is_absolute() or ".." in rel.parts:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid viewer path")
            return
        target = (self.viewer_dir / rel).resolve()
        if not target.is_relative_to(self.viewer_dir.resolve()):
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid viewer path")
            return
        self._serve_file(target, send_body=send_body)

    def _serve_artifact_path(self, request_path: str, *, send_body: bool):
        parts = request_path.split("/", 3)
        if len(parts) < 4:
            self.send_error(HTTPStatus.NOT_FOUND, "Missing artifact path")
            return
        artifact_id = parts[2]
        rel = Path(unquote(parts[3]))
        if rel.is_absolute() or ".." in rel.parts:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid artifact path")
            return
        base_dir = self.artifact_dirs.get(artifact_id)
        if base_dir is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown artifact")
            return
        target = (base_dir / rel).resolve()
        allowed_dirs = [base_dir.resolve(), *self.allowed_roots]
        if not any(target.is_relative_to(root) for root in allowed_dirs):
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid artifact path")
            return
        self._serve_file(target, send_body=send_body)


def main() -> None:
    args = parse_args()
    roots = [Path(root).resolve() for root in args.roots] if args.roots else default_roots()
    artifacts = load_artifacts(roots)
    FSQViewerHandler.artifacts = artifacts
    FSQViewerHandler.artifact_dirs = {
        artifact["id"]: Path(artifact["path"]).resolve() for artifact in artifacts
    }
    FSQViewerHandler.allowed_roots = [root.resolve() for root in roots]
    FSQViewerHandler.viewer_dir = VIEWER_DIR.resolve()

    print("Scanning roots:")
    for root in roots:
        print(f"  {root}")
    print(f"Found {len(artifacts)} FSQ artifacts.")
    print(f"Viewer files: {VIEWER_DIR.resolve()}")
    server = ThreadingHTTPServer((args.host, args.port), FSQViewerHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
