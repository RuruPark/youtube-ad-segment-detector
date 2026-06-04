#!/usr/bin/env python3
'''Serve the current state-machine ad review viewer with manifest media whitelist and Range support.'''
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path('.')
STATIC_ROOT = PROJECT_ROOT / 'outputs/review/state_machine_ad_review_viewer_current'
MANIFEST_PATH = STATIC_ROOT / 'review_manifest_current_train_val.json'
CURRENT_VERSION_PATH = STATIC_ROOT / 'current_version.json'
TEST_VIDEO_IDS = {4, 16, 17}
ALLOWED_STATIC = {
    '/': 'index.html',
    '/index.html': 'index.html',
    '/app.js': 'app.js',
    '/style.css': 'style.css',
    '/review_manifest_current_train_val.json': 'review_manifest_current_train_val.json',
    '/current_version.json': 'current_version.json',
    '/README_current_viewer.md': 'README_current_viewer.md',
}


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_manifest(manifest_path: Path) -> tuple[dict, dict[int, Path]]:
    manifest = load_json(manifest_path)
    scope = str(manifest.get('scope') or '').lower()
    allowed_splits = {'train'} if scope == 'train_only' else {'train', 'validation'}
    whitelist: dict[int, Path] = {}
    for video in manifest.get('videos', []):
        try:
            video_id = int(video.get('video_id'))
        except Exception:
            continue
        split = str(video.get('split') or '').lower()
        if split not in allowed_splits:
            continue
        if video_id in TEST_VIDEO_IDS:
            continue
        video_path_text = str(video.get('video_path') or '')
        if video_path_text:
            whitelist[video_id] = Path(video_path_text)
    return manifest, whitelist


def media_mimetype(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == '.mp4':
        return 'video/mp4'
    if suffix == '.webm':
        return 'video/webm'
    if suffix == '.mov':
        return 'video/quicktime'
    if suffix == '.mkv':
        return 'video/x-matroska'
    if suffix == '.avi':
        return 'video/x-msvideo'
    return 'application/octet-stream'


class CurrentViewerHandler(BaseHTTPRequestHandler):
    server_version = 'StateMachineAdReviewViewerCurrent/1.4'

    @property
    def media_whitelist(self) -> dict[int, Path]:
        return self.server.media_whitelist  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self.handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self.handle_request(send_body=False)

    def handle_request(self, send_body: bool) -> None:
        request_path = unquote(urlparse(self.path).path)
        if request_path.startswith('/media/'):
            self.serve_media(request_path, send_body)
            return
        self.serve_static(request_path, send_body)

    def serve_static(self, request_path: str, send_body: bool) -> None:
        if request_path not in ALLOWED_STATIC:
            self.send_error(HTTPStatus.NOT_FOUND, 'Static file not found')
            return
        static_path = (STATIC_ROOT / ALLOWED_STATIC[request_path]).resolve()
        if not is_relative_to(static_path, STATIC_ROOT) or not static_path.exists() or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, 'Static file not found')
            return
        body = static_path.read_bytes()
        content_type = mimetypes.guess_type(str(static_path))[0] or 'application/octet-stream'
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def serve_media(self, request_path: str, send_body: bool) -> None:
        match = re.fullmatch(r'/media/(\d+)', request_path)
        if not match:
            self.send_error(HTTPStatus.NOT_FOUND, 'Media route must be /media/<video_id>')
            return
        video_id = int(match.group(1))
        if video_id in TEST_VIDEO_IDS:
            self.send_error(HTTPStatus.FORBIDDEN, 'Test video IDs are not served by this review viewer')
            return
        video_path = self.media_whitelist.get(video_id)
        if video_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, 'Video ID is not in the current manifest media whitelist')
            return
        if not video_path.exists() or not video_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, 'Whitelisted video file is missing on the remote server')
            return
        self.send_file_with_range(video_path, send_body)

    def send_file_with_range(self, file_path: Path, send_body: bool) -> None:
        file_size = file_path.stat().st_size
        if file_size <= 0:
            self.send_error(HTTPStatus.NOT_FOUND, 'Whitelisted video file is empty')
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or media_mimetype(file_path)
        range_header = self.headers.get('Range')
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, 'Invalid Range header')
                return
            start_text, end_text = match.groups()
            if start_text == '' and end_text == '':
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, 'Invalid Range header')
                return
            if start_text == '':
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, 'Invalid suffix range')
                    return
                start = max(file_size - suffix_length, 0)
            else:
                start = int(start_text)
            if end_text:
                end = int(end_text)
            if start >= file_size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header('Content-Range', f'bytes */{file_size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                return
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        content_length = end - start + 1
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(content_length))
        self.send_header('Cache-Control', 'no-store')
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.end_headers()
        if not send_body:
            return
        with file_path.open('rb') as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write('%s - - [%s] %s\n' % (self.client_address[0], self.log_date_time_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve current state-machine ad review viewer')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--static-root', default=str(STATIC_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global STATIC_ROOT, MANIFEST_PATH, CURRENT_VERSION_PATH
    STATIC_ROOT = Path(args.static_root).resolve()
    MANIFEST_PATH = STATIC_ROOT / 'review_manifest_current_train_val.json'
    CURRENT_VERSION_PATH = STATIC_ROOT / 'current_version.json'
    if not STATIC_ROOT.exists() or not MANIFEST_PATH.exists():
        print(f'ERROR: current viewer static root or manifest missing: {STATIC_ROOT}', file=sys.stderr)
        return 1
    manifest, whitelist = load_manifest(MANIFEST_PATH)
    if any(video_id in TEST_VIDEO_IDS for video_id in whitelist):
        print('ERROR: test video ID found in media whitelist', file=sys.stderr)
        return 1
    if manifest.get('scope') == 'train_only':
        bad = [video_id for video_id, path in whitelist.items() if video_id not in {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}]
        if bad:
            print(f'ERROR: train-only manifest has non-train media whitelist IDs: {bad}', file=sys.stderr)
            return 1
    current = load_json(CURRENT_VERSION_PATH) if CURRENT_VERSION_PATH.exists() else {}
    server = ThreadingHTTPServer((args.host, args.port), CurrentViewerHandler)
    server.manifest = manifest  # type: ignore[attr-defined]
    server.media_whitelist = whitelist  # type: ignore[attr-defined]
    print('State Machine Ad Review Viewer Current')
    print(f"Current viewer version: {current.get('current_version', manifest.get('viewer_version', manifest.get('version')))}")
    print(f"Detector version: {current.get('detector_version', manifest.get('detector_version'))}")
    print(f"Scope: {current.get('scope', manifest.get('scope'))}")
    print(f'Serving static root: {STATIC_ROOT}')
    print(f'Media whitelist video IDs: {sorted(whitelist)}')
    print(f'Local browser URL after VS Code port forwarding: http://localhost:{args.port}')
    print('Rollback to v1.2: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2')
    print('Switch to v1.3 train-only: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train')
    print('Switch to v1.4 train-only: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_4_train')
    print('Unsupported codecs are not converted by this server.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down current review server')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
