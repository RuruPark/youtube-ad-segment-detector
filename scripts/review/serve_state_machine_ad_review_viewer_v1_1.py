#!/usr/bin/env python3
"""Serve the state-machine ad review viewer v1.1 over HTTP.

The media route serves only train/validation video paths whitelisted by the
generated manifest. It supports HTTP Range requests so browser video seeking can
work without re-encoding or copying media files.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(".")
STATIC_ROOT = PROJECT_ROOT / "outputs/review/state_machine_ad_review_viewer_v1_1"
MANIFEST_PATH = STATIC_ROOT / "review_manifest_v1_1_train_val.json"
EXCLUDED_TEST_VIDEO_IDS = {4, 16, 17}
ALLOWED_STATIC = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
    "/review_manifest_v1_1_train_val.json": "review_manifest_v1_1_train_val.json",
    "/README_review_viewer.md": "README_review_viewer.md",
}


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_manifest(manifest_path: Path) -> tuple[dict, dict[int, Path]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    whitelist: dict[int, Path] = {}
    for video in manifest.get("videos", []):
        video_id = int(video.get("video_id"))
        split = str(video.get("split") or "").lower()
        if split not in {"train", "validation"}:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            continue
        video_path = Path(str(video.get("video_path") or ""))
        if video_path:
            whitelist[video_id] = video_path
    return manifest, whitelist


class ReviewViewerHandler(BaseHTTPRequestHandler):
    server_version = "StateMachineAdReviewViewer/1.1"

    def do_GET(self) -> None:
        self.handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self.handle_request(send_body=False)

    def handle_request(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/media/"):
            self.serve_media(path, send_body)
            return
        self.serve_static(path, send_body)

    @property
    def media_whitelist(self) -> dict[int, Path]:
        return self.server.media_whitelist  # type: ignore[attr-defined]

    def serve_static(self, request_path: str, send_body: bool) -> None:
        if request_path not in ALLOWED_STATIC:
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        relative = Path(ALLOWED_STATIC[request_path])
        static_path = (STATIC_ROOT / relative).resolve()
        if not is_relative_to(static_path, STATIC_ROOT) or not static_path.exists() or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        content_type = mimetypes.guess_type(str(static_path))[0] or "application/octet-stream"
        body = static_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def serve_media(self, request_path: str, send_body: bool) -> None:
        match = re.fullmatch(r"/media/(\d+)", request_path)
        if not match:
            self.send_error(HTTPStatus.NOT_FOUND, "Media route must be /media/<video_id>")
            return
        video_id = int(match.group(1))
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            self.send_error(HTTPStatus.FORBIDDEN, "Test video IDs are not served by this review viewer")
            return
        video_path = self.media_whitelist.get(video_id)
        if video_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Video ID is not in the manifest media whitelist")
            return
        if not video_path.exists() or not video_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Whitelisted video file is missing on the remote server")
            return
        self.send_file_with_range(video_path, send_body)

    def send_file_with_range(self, file_path: Path, send_body: bool) -> None:
        file_size = file_path.stat().st_size
        content_type = mimetypes.guess_type(str(file_path))[0] or media_mimetype(file_path)
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range header")
                return
            start_text, end_text = match.groups()
            if start_text == "" and end_text == "":
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range header")
                return
            if start_text == "":
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid suffix range")
                    return
                start = max(file_size - suffix_length, 0)
            else:
                start = int(start_text)
            if end_text:
                end = int(end_text)
            if start >= file_size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if not send_body:
            return
        with file_path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))


def media_mimetype(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".mkv":
        return "video/x-matroska"
    return "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve state-machine ad review viewer v1.1")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default keeps the server local for VS Code port forwarding.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--static-root", default=str(STATIC_ROOT), help="Static viewer directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global STATIC_ROOT, MANIFEST_PATH
    STATIC_ROOT = Path(args.static_root).resolve()
    MANIFEST_PATH = STATIC_ROOT / "review_manifest_v1_1_train_val.json"
    if not STATIC_ROOT.exists():
        print(f"ERROR: static root does not exist: {STATIC_ROOT}", file=sys.stderr)
        return 1
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest does not exist: {MANIFEST_PATH}", file=sys.stderr)
        return 1
    manifest, media_whitelist = load_manifest(MANIFEST_PATH)
    if any(video_id in EXCLUDED_TEST_VIDEO_IDS for video_id in media_whitelist):
        print("ERROR: test video ID found in media whitelist", file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((args.host, args.port), ReviewViewerHandler)
    server.manifest = manifest  # type: ignore[attr-defined]
    server.media_whitelist = media_whitelist  # type: ignore[attr-defined]
    print("State Machine Ad Review Viewer v1.1")
    print(f"Serving static root: {STATIC_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Media whitelist video IDs: {sorted(media_whitelist)}")
    print(f"Open from VS Code Remote-SSH: forward port {args.port} in the Ports panel")
    print(f"Local browser URL after forwarding: http://localhost:{args.port}")
    print("Unsupported codecs are not converted by this server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down review server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
