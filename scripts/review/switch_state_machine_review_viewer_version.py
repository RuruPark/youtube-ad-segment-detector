#!/usr/bin/env python3
'''Switch the current state-machine review viewer between supported versions.'''
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path('.')
REGISTRY_PATH = PROJECT_ROOT / 'outputs/review/state_machine_ad_review_viewer_versions.json'
CURRENT_MANIFEST_NAME = 'review_manifest_current_train_val.json'
SUPPORTED_VERSIONS = {'v1_2', 'v1_3_train', 'v1_4_train'}
TEST_VIDEO_IDS = {4, 16, 17}
VALIDATION_VIDEO_IDS = {3, 7, 18}
FORBIDDEN_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.parquet', '.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.onnx'}
FORBIDDEN_DIRECTORY_PARTS = {'cache', 'frames', 'frame_images', 'raw_video', 'video_proxy', 'model_cache', 'tmp', '__pycache__'}


def now_stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def scan_forbidden(root: Path) -> list[str]:
    bad: list[str] = []
    if not root.exists():
        return bad
    for path in root.rglob('*'):
        rel_parts = {part.lower() for part in path.relative_to(root).parts}
        if rel_parts & FORBIDDEN_DIRECTORY_PARTS:
            bad.append(str(path.relative_to(root)))
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(str(path.relative_to(root)))
    return sorted(bad)


def patch_app_for_current(app_text: str, source_manifest_name: str) -> str:
    manifest_names = {
        source_manifest_name,
        'review_manifest_v1_1_train_val.json',
        'review_manifest_v1_2_train_val.json',
        'review_manifest_v1_3_train.json',
        'review_manifest_v1_4_train.json',
    }
    for name in manifest_names:
        app_text = app_text.replace(name, CURRENT_MANIFEST_NAME)
    return app_text


def manifest_video_ids(manifest: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for video in manifest.get('videos', []):
        try:
            ids.add(int(video.get('video_id')))
        except Exception:
            continue
    return ids


def validate_manifest_for_version(version: str, manifest: dict[str, Any]) -> None:
    ids = manifest_video_ids(manifest)
    test_ids = ids & TEST_VIDEO_IDS
    if test_ids or manifest.get('split_policy', {}).get('test_included'):
        raise RuntimeError(f'test video IDs found in current manifest: {sorted(test_ids)}')
    if version in {'v1_3_train', 'v1_4_train'}:
        validation_ids = ids & VALIDATION_VIDEO_IDS
        splits = {str(video.get('split') or '').lower() for video in manifest.get('videos', [])}
        if validation_ids or splits - {'train'} or manifest.get('split_policy', {}).get('validation_included'):
            raise RuntimeError(f'{version} current manifest must be train-only; validation_ids={sorted(validation_ids)}, splits={sorted(splits)}')
        expected_detector = 'v1.4' if version == 'v1_4_train' else 'v1.3'
        if manifest.get('detector_version') != expected_detector or manifest.get('scope') != 'train_only':
            raise RuntimeError(f'{version} manifest must declare detector_version={expected_detector} and scope=train_only')


def write_current_readme(current_dir: Path, version: str, info: dict[str, Any], manifest: dict[str, Any]) -> None:
    scope = info.get('scope') or manifest.get('scope') or 'unknown'
    detector_version = info.get('detector_version') or manifest.get('detector_version') or version
    text = f'''# Current State Machine Ad Review Viewer

Current viewer version: `{version}` ({detector_version}, {scope}).

## Run From VS Code Remote-SSH

```bash
cd .
python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000
```

Forward port `8000` in VS Code Ports panel and open `http://localhost:8000` locally.

## Rollback / Switch

Rollback to v1.2:

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2
```

Switch current viewer to v1.3 train-only:

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train
```

No media files are copied. The server serves only video paths whitelisted by `{CURRENT_MANIFEST_NAME}`. Unsupported codecs are not converted.
'''
    current_dir.joinpath('README_current_viewer.md').write_text(text, encoding='utf-8')


def current_version_payload(version: str, info: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    split_policy = manifest.get('split_policy', {})
    scope = info.get('scope') or manifest.get('scope')
    detector_version = info.get('detector_version') or manifest.get('detector_version')
    payload = {
        'current_version': version,
        'detector_version': detector_version,
        'base_version': info.get('base_version') or manifest.get('base_version'),
        'scope': scope,
        'rollback_supported': True,
        'rollback_target': 'v1_3_train' if version == 'v1_4_train' else ('v1_2' if version == 'v1_3_train' else 'v1_3_train'),
        'validation_included': bool(split_policy.get('validation_included')),
        'test_included': bool(split_policy.get('test_included')),
        'updated_at': now_iso(),
    }
    if version in {'v1_3_train', 'v1_4_train'}:
        payload['validation_included'] = False
        payload['test_included'] = False
    return payload


def switch_version(version: str, project_root: Path, dry_run: bool) -> dict[str, Any]:
    registry_path = project_root / REGISTRY_PATH.relative_to(PROJECT_ROOT)
    if not registry_path.exists():
        raise FileNotFoundError(f'version registry missing: {registry_path}')
    registry = read_json(registry_path)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f'unsupported version {version}; supported={sorted(SUPPORTED_VERSIONS)}')
    available = registry.get('available_versions', {})
    if version not in available:
        raise ValueError(f'version {version} not registered; registered={sorted(available)}')
    info = dict(available[version])
    source_dir = project_root / info['viewer_dir']
    current_dir = project_root / registry.get('current_viewer_dir', 'outputs/review/state_machine_ad_review_viewer_current')
    manifest_name = info.get('manifest')
    if not manifest_name:
        manifest_name = 'review_manifest_v1_2_train_val.json' if version == 'v1_2' else 'review_manifest_v1_3_train.json'
    required = ['index.html', 'app.js', 'style.css', manifest_name]
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'missing required file(s) in {source_dir}: {missing}')
    source_manifest = read_json(source_dir / manifest_name)
    validate_manifest_for_version(version, source_manifest)
    backup_dir = project_root / 'backups' / f'state_machine_ad_review_viewer_current_switch_{version}_{now_stamp()}' if current_dir.exists() else None
    actions = {
        'version': version,
        'source_dir': str(source_dir),
        'source_manifest': manifest_name,
        'current_dir': str(current_dir),
        'backup_dir': str(backup_dir) if backup_dir else None,
        'dry_run': dry_run,
    }
    if dry_run:
        return actions
    if backup_dir:
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(current_dir, backup_dir)
        shutil.rmtree(current_dir)
    current_dir.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'style.css']:
        shutil.copy2(source_dir / name, current_dir / name)
    app_text = (source_dir / 'app.js').read_text(encoding='utf-8')
    (current_dir / 'app.js').write_text(patch_app_for_current(app_text, manifest_name), encoding='utf-8')
    shutil.copy2(source_dir / manifest_name, current_dir / CURRENT_MANIFEST_NAME)
    payload = current_version_payload(version, info, source_manifest)
    write_json(current_dir / 'current_version.json', payload)
    write_current_readme(current_dir, version, info, source_manifest)
    registry['current_version'] = version
    registry['rollback_supported'] = True
    write_json(registry_path, registry)
    bad = scan_forbidden(current_dir)
    if bad:
        raise RuntimeError(f'forbidden files found in current viewer: {bad}')
    current_manifest = read_json(current_dir / CURRENT_MANIFEST_NAME)
    validate_manifest_for_version(version, current_manifest)
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Switch current state-machine review viewer version')
    parser.add_argument('--version', required=True, choices=sorted(SUPPORTED_VERSIONS))
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if project_root != PROJECT_ROOT:
        print(f'ERROR: project root must be {PROJECT_ROOT}, got {project_root}', file=sys.stderr)
        return 2
    actions = switch_version(args.version, project_root, args.dry_run)
    print(f"Selected viewer version: {args.version}")
    print(f"Source: {actions['source_dir']}")
    print(f"Source manifest: {actions['source_manifest']}")
    print(f"Current viewer: {actions['current_dir']}")
    if actions.get('backup_dir'):
        print(f"Backup: {actions['backup_dir']}")
    if args.dry_run:
        print('Dry run only; no files changed.')
    else:
        print('Switch complete.')
        print('Run: python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000')
        print('Then forward port 8000 in VS Code and open http://localhost:8000')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
