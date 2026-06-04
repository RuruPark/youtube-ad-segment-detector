#!/usr/bin/env python3
"""샘플 광고 스킵 viewer를 manifest 기반 media whitelist로 제공한다."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
STATIC_ROOT = PROJECT_ROOT / "outputs/demo/final_presentation_ad_skip_viewer"
MANIFEST_PATH = STATIC_ROOT / "demo_viewer_manifest.json"
ALLOWED_STATIC = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
    "/demo_viewer_manifest.json": "demo_viewer_manifest.json",
    "/demo_viewer_manifest.js": "demo_viewer_manifest.js",
    "/demo_viewer_metrics.json": "demo_viewer_metrics.json",
    "/demo_viewer_metrics.js": "demo_viewer_metrics.js",
}


def is_relative_to(path: Path, parent: Path) -> bool:
    """경로가 허용된 상위 폴더 안에 있는지 확인한다."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_manifest(manifest_path: Path) -> tuple[dict, dict[int, Path]]:
    """데모 manifest를 읽고 로컬 재생에 허용할 media 목록을 만든다."""
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    whitelist: dict[int, Path] = {}
    allowed_ids = {int(video_id) for video_id in manifest.get("test_video_ids", [])}
    for video in manifest.get("videos", []):
        try:
            video_id = int(video.get("video_id"))
        except Exception:
            continue
        if allowed_ids and video_id not in allowed_ids:
            continue
        video_path_text = str(video.get("video_path") or "").strip()
        if not video_path_text or video.get("playable") is False:
            continue
        video_path = Path(video_path_text)
        if not video_path.is_absolute():
            video_path = (PROJECT_ROOT / video_path).resolve()
        whitelist[video_id] = video_path
    return manifest, whitelist


def media_mimetype(path: Path) -> str:
    """브라우저가 해석하기 쉬운 video MIME type을 반환한다."""
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".mkv":
        return "video/x-matroska"
    if suffix == ".avi":
        return "video/x-msvideo"
    return "application/octet-stream"


class DemoViewerHandler(BaseHTTPRequestHandler):
    server_version = "PublicAdSkipDemoViewer/1.0"

    @property
    def media_whitelist(self) -> dict[int, Path]:
        return self.server.media_whitelist  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self.handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self.handle_request(send_body=False)

    def handle_request(self, send_body: bool) -> None:
        request_path = unquote(urlparse(self.path).path)
        if request_path.startswith("/media/"):
            self.serve_media(request_path, send_body)
            return
        self.serve_static(request_path, send_body)

    def serve_static(self, request_path: str, send_body: bool) -> None:
        if request_path not in ALLOWED_STATIC:
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        static_path = (STATIC_ROOT / ALLOWED_STATIC[request_path]).resolve()
        if not is_relative_to(static_path, STATIC_ROOT) or not static_path.exists() or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        body = static_path.read_bytes()
        content_type = mimetypes.guess_type(str(static_path))[0] or "application/octet-stream"
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
        video_path = self.media_whitelist.get(video_id)
        if video_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Video ID is not in the demo media whitelist")
            return
        if not video_path.exists() or not video_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Whitelisted video file is missing")
            return
        self.send_file_with_range(video_path, send_body)

    def send_file_with_range(self, file_path: Path, send_body: bool) -> None:
        file_size = file_path.stat().st_size
        if file_size <= 0:
            self.send_error(HTTPStatus.NOT_FOUND, "Whitelisted video file is empty")
            return
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve final presentation ad-skip demo viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--static-root", default=str(STATIC_ROOT))
    parser.add_argument("--open", action="store_true", help="Open the local URL in the default browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global STATIC_ROOT, MANIFEST_PATH
    STATIC_ROOT = Path(args.static_root).resolve()
    MANIFEST_PATH = STATIC_ROOT / "demo_viewer_manifest.json"
    if not STATIC_ROOT.exists() or not MANIFEST_PATH.exists():
        print(f"ERROR: demo viewer static root or manifest missing: {STATIC_ROOT}", file=sys.stderr)
        return 1
    manifest, whitelist = load_manifest(MANIFEST_PATH)
    missing = [video_id for video_id, path in whitelist.items() if not path.exists()]
    if missing:
        print(f"ERROR: missing whitelisted video files for IDs: {missing}", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((args.host, args.port), DemoViewerHandler)
    server.manifest = manifest  # type: ignore[attr-defined]
    server.media_whitelist = whitelist  # type: ignore[attr-defined]
    url = f"http://localhost:{args.port}"
    print("Public Ad Skip Demo Viewer")
    print(f"Serving static root: {STATIC_ROOT}")
    print(f"Media whitelist video IDs: {sorted(whitelist)}")
    print(f"Local browser URL: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo viewer server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
