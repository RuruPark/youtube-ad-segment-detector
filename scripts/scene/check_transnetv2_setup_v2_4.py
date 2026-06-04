#!/usr/bin/env python3
"""Check project-local TransNetV2 PyTorch setup for v2.4 scene experiments.

This script verifies whether TransNetV2 can be used in the existing cv conda
runtime without modifying detector assets, labels, splits, visual anchors, or
prediction outputs. It may install the PyPI package into a project-local target
when explicitly requested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
TASK_NAME = 'transnetv2_setup_check_v2_4'
VERSION = 'v2_4'
CONDA_EXE = 'conda'
CONDA_ENV = 'cv'
CONDA_PYTHON = Path('.venv/bin/python')
CONDA_FFMPEG = Path('.venv/bin/ffmpeg')
PACKAGE_NAME = 'transnetv2-pytorch'
PACKAGE_VERSION_PIN = '1.0.5'
TARGET_DIR = PROJECT_ROOT / 'external' / 'transnetv2' / 'python'
WEIGHT_BASENAME = 'transnetv2-pytorch-weights.pth'
MODEL_DIR = PROJECT_ROOT / 'models' / 'third_party' / 'transnetv2'
SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'scene' / 'check_transnetv2_setup_v2_4.py'
INVENTORY_CSV = PROJECT_ROOT / 'data' / 'scene' / 'transnetv2_model_inventory_v2_4.csv'
SUMMARY_MD = PROJECT_ROOT / 'reports' / 'scene' / 'transnetv2_setup_check_v2_4_summary.md'
REPORT_JSON = PROJECT_ROOT / 'reports' / 'scene' / 'transnetv2_setup_check_v2_4_report.json'
RUN_LOG = PROJECT_ROOT / 'logs' / 'transnetv2_setup_check_v2_4_run_log.txt'
LATEST_DIR = PROJECT_ROOT / 'outputs' / 'latest_for_chatgpt_transnetv2_setup_check_v2_4'
LATEST_README = LATEST_DIR / 'README_latest_files.md'

REFERENCE_SOURCES = [
    {
        'name': 'PyPI transnetv2-pytorch',
        'url': 'https://pypi.org/project/transnetv2-pytorch/',
        'notes': 'Package 1.0.5, Python >=3.10, CLI and Python API; wheel includes bundled PyTorch weights.',
    },
    {
        'name': 'Official TransNetV2 GitHub',
        'url': 'https://github.com/soCzech/TransNetV2',
        'notes': 'Original TransNet V2 repository documents inference use without retraining and links PyTorch inference code.',
    },
    {
        'name': 'HuggingFace TransNetV2 PyTorch weights reference',
        'url': 'https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights',
        'notes': 'Fallback public PyTorch weight source; not used because PyPI wheel bundled weights.',
    },
]

PROTECTED_PATHS = [
    PROJECT_ROOT / 'scripts' / 'detectors',
    PROJECT_ROOT / 'configs' / 'detectors',
    PROJECT_ROOT / 'data' / 'predictions',
    PROJECT_ROOT / 'reports' / 'detectors',
    PROJECT_ROOT / 'data' / 'features' / 'visual_scene_boundary_anchors_v2_4.csv',
    PROJECT_ROOT / 'data' / 'features' / 'visual_scene_boundary_anchors_v2_4_with_split.csv',
    PROJECT_ROOT / 'data' / 'segments' / 'ad_interval_segments_v2_4.csv',
    PROJECT_ROOT / 'data' / 'splits' / 'video_split_v2_4.csv',
    OLD_PROJECT_ROOT,
]

FORBIDDEN_LATEST_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.pt', '.pth', '.ckpt', '.bin'}
FORBIDDEN_LATEST_TOKENS = {'raw', 'frame', 'frames', 'cache', 'checkpoint', 'weights'}
LOG_LINES: list[str] = []


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def log(message: str) -> None:
    line = f'{now_iso()} {message}'
    LOG_LINES.append(line)
    print(message, flush=True)


def write_log() -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')


def run_cmd(cmd: list[str], env_extra: dict[str, str] | None = None, timeout: int = 180) -> dict[str, Any]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, capture_output=True, timeout=timeout)
        return {
            'cmd': cmd,
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'seconds': round(time.time() - started, 3),
        }
    except Exception as exc:
        return {'cmd': cmd, 'returncode': -999, 'stdout': '', 'stderr': repr(exc), 'seconds': round(time.time() - started, 3)}


def conda_python_cmd(args: list[str]) -> list[str]:
    return [CONDA_EXE, 'run', '-n', CONDA_ENV, 'python', *args]


def py_env() -> dict[str, str]:
    existing = os.environ.get('PYTHONPATH', '')
    paths = [str(TARGET_DIR)]
    if existing:
        paths.append(existing)
    return {'PYTHONPATH': os.pathsep.join(paths)}


def hash_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'exists': False, 'file_count': 0, 'total_size': 0, 'signature': ''}
    if path.is_file():
        stat = path.stat()
        payload = f'{path.name}|{stat.st_size}|{stat.st_mtime_ns}'
        return {'exists': True, 'file_count': 1, 'total_size': stat.st_size, 'signature': hashlib.sha256(payload.encode()).hexdigest()}
    digest = hashlib.sha256()
    count = 0
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(dirs)
        for name in sorted(files):
            full = Path(root) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            rel = full.relative_to(path)
            digest.update(f'{rel}|{stat.st_size}|{stat.st_mtime_ns}\n'.encode('utf-8', 'ignore'))
            count += 1
            total += stat.st_size
    return {'exists': True, 'file_count': count, 'total_size': total, 'signature': digest.hexdigest()}


def protected_snapshot() -> dict[str, Any]:
    return {str(path): tree_signature(path) for path in PROTECTED_PATHS}


def format_cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({col: format_cell(row.get(col, '')) for col in columns})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def parse_key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            out[key.strip()] = value.strip()
    return out


def inspect_env() -> dict[str, Any]:
    py_version = run_cmd(conda_python_cmd(['--version']))
    py_exe = run_cmd(conda_python_cmd(['-c', 'import sys; print(sys.executable)']))
    pip_ver = run_cmd(conda_python_cmd(['-m', 'pip', '--version']))
    torch = run_cmd(conda_python_cmd(['-c', "import torch; print('torch_version=' + torch.__version__); print('torch_file=' + torch.__file__); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_version=' + str(getattr(torch.version, 'cuda', None)))"]))
    ffmpeg = run_cmd([str(CONDA_FFMPEG), '-version']) if CONDA_FFMPEG.exists() else {'returncode': 127, 'stdout': '', 'stderr': 'ffmpeg not found', 'cmd': [str(CONDA_FFMPEG)]}
    torch_kv = parse_key_values(torch['stdout'])
    return {
        'python_executable': py_exe['stdout'].strip(),
        'python_version': py_version['stdout'].strip() or py_version['stderr'].strip(),
        'pip_version': pip_ver['stdout'].strip(),
        'torch_version': torch_kv.get('torch_version', ''),
        'torch_file': torch_kv.get('torch_file', ''),
        'cuda_available': torch_kv.get('cuda_available', '').lower() == 'true',
        'cuda_version': torch_kv.get('cuda_version', ''),
        'ffmpeg_available': ffmpeg['returncode'] == 0,
        'ffmpeg_path': str(CONDA_FFMPEG) if CONDA_FFMPEG.exists() else '',
        'ffmpeg_version_summary': ffmpeg['stdout'].splitlines()[0] if ffmpeg['stdout'] else ffmpeg['stderr'].splitlines()[0] if ffmpeg['stderr'] else '',
        'commands': {'python': py_version, 'pip': pip_ver, 'torch': torch, 'ffmpeg': ffmpeg},
    }


def check_package_import() -> dict[str, Any]:
    code = (
        "import os, importlib.metadata as md, importlib.util; "
        "spec=importlib.util.find_spec('transnetv2_pytorch'); print('spec=' + (spec.origin if spec else '')); "
        "import transnetv2_pytorch; print('version=' + md.version('transnetv2-pytorch')); "
        "print('module_path=' + transnetv2_pytorch.__file__); "
        "print('weight_path=' + os.path.join(os.path.dirname(transnetv2_pytorch.__file__), 'transnetv2-pytorch-weights.pth'))"
    )
    result = run_cmd(conda_python_cmd(['-c', code]), env_extra=py_env(), timeout=120)
    kv = parse_key_values(result['stdout'])
    return {
        'available': result['returncode'] == 0 and bool(kv.get('module_path')),
        'version': kv.get('version', ''),
        'module_path': kv.get('module_path', ''),
        'weight_path': kv.get('weight_path', ''),
        'command': result,
    }


def install_project_local(force: bool, warnings: list[str]) -> dict[str, Any]:
    before = check_package_import()
    if before['available'] and not force:
        return {'performed': False, 'method': 'existing project-local package reused', 'before': before, 'dry_run': {}, 'install': {}, 'removed_existing_target': False}
    removed = False
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
        removed = True
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    dry = run_cmd(conda_python_cmd(['-m', 'pip', 'install', '--dry-run', f'{PACKAGE_NAME}=={PACKAGE_VERSION_PIN}']), timeout=240)
    if 'torch-2.12.0' in (dry['stdout'] + dry['stderr']) or 'cuda-toolkit==13' in (dry['stdout'] + dry['stderr']):
        warnings.append('pip dry-run shows resolver may choose torch 2.12/CUDA 13 if dependencies are installed into target; final install used --no-deps to reuse cv torch 2.11.')
    install = run_cmd(
        conda_python_cmd([
            '-m', 'pip', 'install', '--no-deps', '--target', str(TARGET_DIR),
            f'{PACKAGE_NAME}=={PACKAGE_VERSION_PIN}', 'ffmpeg-python==0.2.0', 'future==1.0.0',
        ]),
        timeout=600,
    )
    return {
        'performed': install['returncode'] == 0,
        'method': f'project-local pip --target {TARGET_DIR} --no-deps using conda env {CONDA_ENV}; existing cv torch reused',
        'before': before,
        'dry_run': dry,
        'install': install,
        'removed_existing_target': removed,
    }


def verify_weight(weight_path_text: str) -> dict[str, Any]:
    path = Path(weight_path_text) if weight_path_text else TARGET_DIR / 'transnetv2_pytorch' / WEIGHT_BASENAME
    exists = path.exists()
    return {
        'available': exists,
        'path': str(path) if exists else str(path),
        'size_bytes': path.stat().st_size if exists else 0,
        'sha256': hash_file(path) if exists else '',
        'source_url_or_package': f'{PACKAGE_NAME}=={PACKAGE_VERSION_PIN} bundled wheel from PyPI',
        'separate_weight_download_performed': False,
    }


def run_cli_help() -> dict[str, Any]:
    return run_cmd(conda_python_cmd(['-m', 'transnetv2_pytorch', '--help']), env_extra=py_env(), timeout=120)


def run_smoke_test() -> dict[str, Any]:
    code = (
        "import os, time, torch, transnetv2_pytorch; from transnetv2_pytorch import TransNetV2; "
        "start=time.time(); model=TransNetV2(device='cpu'); model.eval(); "
        "frames=torch.zeros((1,100,27,48,3), dtype=torch.uint8); "
        "torch.set_grad_enabled(False); one_hot, extra = model(frames); "
        "print('model_device=' + str(model.device)); "
        "print('model_init_seconds=' + str(round(time.time()-start, 3))); "
        "print('one_hot_shape=' + str(tuple(one_hot.shape))); "
        "print('many_hot_shape=' + str(tuple(extra['many_hot'].shape))); "
        "print('forward_ok=True')"
    )
    result = run_cmd(conda_python_cmd(['-c', code]), env_extra=py_env(), timeout=240)
    kv = parse_key_values(result['stdout'])
    return {
        'status': 'PASS' if result['returncode'] == 0 and kv.get('forward_ok') == 'True' else 'FAIL',
        'command': 'PYTHONPATH={target} conda run -n cv python -c "import TransNetV2; model=TransNetV2(device=cpu); dummy forward"'.format(target=TARGET_DIR),
        'output_summary': kv,
        'raw_result': result,
    }


def build_inventory(env: dict[str, Any], package: dict[str, Any], weight: dict[str, Any], cli: dict[str, Any], smoke: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {'item_type': 'python', 'name': 'conda cv python', 'status': 'available', 'version': env['python_version'], 'path': env['python_executable'], 'size_bytes': '', 'sha256': '', 'source_url_or_package': 'local conda env cv', 'notes': ''},
        {'item_type': 'pip', 'name': 'pip', 'status': 'available' if env['pip_version'] else 'missing', 'version': env['pip_version'], 'path': env['python_executable'], 'size_bytes': '', 'sha256': '', 'source_url_or_package': 'local conda env cv', 'notes': ''},
        {'item_type': 'torch', 'name': 'torch', 'status': 'available' if env['torch_version'] else 'missing', 'version': env['torch_version'], 'path': env.get('torch_file', ''), 'size_bytes': '', 'sha256': '', 'source_url_or_package': 'existing cv env; not upgraded', 'notes': f"cuda_available={env['cuda_available']} cuda_version={env['cuda_version']}"},
        {'item_type': 'cuda', 'name': 'cuda', 'status': 'available' if env['cuda_available'] else 'unavailable', 'version': env['cuda_version'], 'path': '', 'size_bytes': '', 'sha256': '', 'source_url_or_package': 'torch.cuda', 'notes': ''},
        {'item_type': 'ffmpeg', 'name': 'ffmpeg', 'status': 'available' if env['ffmpeg_available'] else 'missing', 'version': env['ffmpeg_version_summary'], 'path': env['ffmpeg_path'], 'size_bytes': '', 'sha256': '', 'source_url_or_package': 'existing cv env', 'notes': ''},
        {'item_type': 'package', 'name': PACKAGE_NAME, 'status': 'available' if package['available'] else 'missing', 'version': package.get('version', ''), 'path': package.get('module_path', ''), 'size_bytes': '', 'sha256': '', 'source_url_or_package': 'https://pypi.org/project/transnetv2-pytorch/', 'notes': f'PYTHONPATH={TARGET_DIR}'},
        {'item_type': 'weight', 'name': WEIGHT_BASENAME, 'status': 'available' if weight['available'] else 'missing', 'version': package.get('version', ''), 'path': weight['path'], 'size_bytes': weight['size_bytes'], 'sha256': weight['sha256'], 'source_url_or_package': weight['source_url_or_package'], 'notes': 'bundled in package; not copied to latest bundle'},
        {'item_type': 'cli', 'name': 'python -m transnetv2_pytorch --help', 'status': 'available' if cli['returncode'] == 0 else 'failed', 'version': package.get('version', ''), 'path': str(TARGET_DIR / 'bin' / 'transnetv2_pytorch'), 'size_bytes': '', 'sha256': '', 'source_url_or_package': PACKAGE_NAME, 'notes': (cli['stdout'] or cli['stderr']).splitlines()[0] if (cli['stdout'] or cli['stderr']) else ''},
        {'item_type': 'smoke_test', 'name': 'cpu_import_weight_load_dummy_forward', 'status': smoke['status'], 'version': package.get('version', ''), 'path': SCRIPT_PATH, 'size_bytes': '', 'sha256': '', 'source_url_or_package': PACKAGE_NAME, 'notes': json.dumps(smoke['output_summary'], ensure_ascii=False)},
    ]


def latest_bundle(paths: list[Path], warnings: list[str]) -> dict[str, Any]:
    if LATEST_DIR.exists():
        for child in LATEST_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    forbidden = []
    for path in paths:
        if not path.exists():
            warnings.append(f'latest source missing: {path}')
            continue
        target = LATEST_DIR / path.name
        shutil.copy2(path, target)
        copied.append(target)
    readme = '# Latest Files: TransNetV2 Setup Check v2.4\n\nThis bundle intentionally excludes model weights, package directories, raw videos, frames, and caches.\n\n## Included\n'
    for path in copied:
        readme += f'- `{path.name}`\n'
    readme += '- `README_latest_files.md`\n'
    LATEST_README.write_text(readme, encoding='utf-8')
    copied.append(LATEST_README)
    for path in copied:
        lower = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES or any(token in lower for token in FORBIDDEN_LATEST_TOKENS):
            forbidden.append(str(path))
    return {'path': str(LATEST_DIR), 'files': [str(path) for path in copied], 'forbidden_files': forbidden}


def make_summary(report: dict[str, Any]) -> str:
    command_example = (
        f"PYTHONPATH={TARGET_DIR} conda run -n cv python -m transnetv2_pytorch "
        f"/path/to/video.mp4 --device cpu --format csv --output /path/to/output.csv"
    )
    return f"""# TransNetV2 Setup Check v2.4

## 1. Why TransNetV2
TransNetV2 is a shot-boundary detection model. For this project it is a candidate third scene-transition source to test later, not a detector rule and not an ad classifier.

## 2. Environment
- Python: `{report['python_version']}` at `{report['python_executable']}`
- pip: `{report['pip_version']}`
- torch: `{report['torch_version']}`
- CUDA available: `{report['cuda_available']}` (`{report['cuda_version']}`)
- ffmpeg available: `{report['ffmpeg_available']}` at `{report['ffmpeg_path']}`

## 3. Install/Download
- install_performed: `{report['install_performed']}`
- install_method: `{report['install_method']}`
- download_performed: `{report['download_performed']}`
- venv_used: `{report['venv_used']}`
- Existing `cv` torch/CUDA environment was not upgraded.

## 4. Package
- package: `{report['transnetv2_package_name']}`
- version: `{report['transnetv2_package_version']}`
- module path: `{report['transnetv2_module_path']}`
- CLI available: `{report['transnetv2_cli_available']}`

## 5. Weight
- available: `{report['weight_available']}`
- path: `{report['weight_path']}`
- size_bytes: `{report['weight_size_bytes']}`
- sha256: `{report['weight_sha256']}`
- The weight is bundled in the PyPI wheel; no separate HuggingFace/GitHub weight download was needed.

## 6. Smoke Test
- status: `{report['smoke_test_status']}`
- command: `{report['smoke_test_command']}`
- output summary: `{report['smoke_test_output_summary']}`

## 7. Next Command Example
```bash
{command_example}
```
For the next extraction step, point this at train videos only and write a new train-only TransNetV2 candidate output. This setup check did not process the train dataset.

## 8. Safety
- protected_files_modified: `{report['protected_files_modified']}`
- latest bundle excludes weight/raw/cache/model files: `{not report.get('latest_bundle_forbidden_files', [])}`
- validation/test row-level output generated: `false`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description='Check TransNetV2 PyTorch setup for v2.4 project')
    parser.add_argument('--force-reinstall-local', action='store_true', help='Remove and reinstall project-local TransNetV2 target with --no-deps')
    parser.add_argument('--install-if-missing', action='store_true', help='Install project-local TransNetV2 target if import is missing')
    args = parser.parse_args()

    start = time.time()
    warnings: list[str] = []
    errors: list[Any] = []
    for d in [TARGET_DIR.parent, MODEL_DIR, INVENTORY_CSV.parent, SUMMARY_MD.parent, REPORT_JSON.parent, RUN_LOG.parent, LATEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        log('[STEP 01] Safety snapshot and output path preparation')
        before = protected_snapshot()

        log('[STEP 02] Inspect current Python, pip, torch, CUDA, ffmpeg environment')
        env = inspect_env()

        log('[STEP 03] Search existing TransNetV2 packages and weights')
        package_before = check_package_import()
        local_weight_before = verify_weight(package_before.get('weight_path', ''))

        log('[STEP 04] Decide safe installation/download strategy')
        do_install = args.force_reinstall_local or (args.install_if_missing and not package_before['available'])
        strategy = 'project-local --target --no-deps install; reuse existing cv torch' if do_install else 'reuse existing project-local package'
        log(f'strategy={strategy}')

        log('[STEP 05] Install or download TransNetV2 package/weights if needed')
        install_info = install_project_local(force=args.force_reinstall_local, warnings=warnings) if do_install else {'performed': False, 'method': 'existing project-local package reused', 'before': package_before, 'dry_run': {}, 'install': {}, 'removed_existing_target': False}

        log('[STEP 06] Verify package import and version')
        package = check_package_import()
        log(f"package_available={package['available']} version={package.get('version', '')} module={package.get('module_path', '')}")

        log('[STEP 07] Verify weight file existence, size, and sha256')
        weight = verify_weight(package.get('weight_path', ''))
        log(f"weight_available={weight['available']} path={weight['path']} size={weight['size_bytes']} sha256={weight['sha256']}")

        log('[STEP 08] Run minimal smoke test without processing full dataset')
        cli = run_cli_help()
        smoke = run_smoke_test()
        log(f"cli_help_returncode={cli['returncode']} smoke_test_status={smoke['status']}")

        log('[STEP 09] Generate model inventory, report, and log')
        after = protected_snapshot()
        protected_modified = {path: {'before': before.get(path), 'after': after.get(path)} for path in before if before.get(path) != after.get(path)}
        setup_ready = package['available'] and weight['available'] and cli['returncode'] == 0 and smoke['status'] == 'PASS'
        setup_status = 'READY_FOR_EXTRACTION_WITH_EXISTING_PROJECT_VIDEOS' if setup_ready else 'NOT_READY'
        inventory = build_inventory(env, package, weight, cli, smoke)
        columns = ['item_type', 'name', 'status', 'version', 'path', 'size_bytes', 'sha256', 'source_url_or_package', 'notes']
        write_csv(INVENTORY_CSV, inventory, columns)

        report = {
            'task_name': TASK_NAME,
            'project_root': str(PROJECT_ROOT),
            'setup_status': setup_status,
            'python_executable': env['python_executable'],
            'python_version': env['python_version'],
            'pip_version': env['pip_version'],
            'torch_version': env['torch_version'],
            'torch_file': env['torch_file'],
            'cuda_available': env['cuda_available'],
            'cuda_version': env['cuda_version'],
            'ffmpeg_available': env['ffmpeg_available'],
            'ffmpeg_path': env['ffmpeg_path'],
            'transnetv2_package_available': package['available'],
            'transnetv2_package_name': PACKAGE_NAME,
            'transnetv2_package_version': package.get('version', ''),
            'transnetv2_module_path': package.get('module_path', ''),
            'transnetv2_cli_available': cli['returncode'] == 0,
            'weight_available': weight['available'],
            'weight_path': weight['path'],
            'weight_size_bytes': weight['size_bytes'],
            'weight_sha256': weight['sha256'],
            'download_performed': bool(install_info.get('performed')),
            'download_sources': REFERENCE_SOURCES,
            'install_performed': bool(install_info.get('performed')),
            'install_method': install_info.get('method', ''),
            'install_removed_existing_target': install_info.get('removed_existing_target', False),
            'install_dry_run_returncode': install_info.get('dry_run', {}).get('returncode', ''),
            'install_returncode': install_info.get('install', {}).get('returncode', ''),
            'venv_used': False,
            'venv_path': '',
            'project_local_package_path': str(TARGET_DIR),
            'smoke_test_status': smoke['status'],
            'smoke_test_command': smoke['command'],
            'smoke_test_output_summary': smoke['output_summary'],
            'cli_help_output_summary': (cli['stdout'] or cli['stderr'])[:2000],
            'ready_for_next_extraction_step': setup_ready,
            'warnings': warnings,
            'errors': errors,
            'protected_files_modified': protected_modified,
            'validation_test_row_level_output_generated': False,
            'full_train_inference_run': False,
            'raw_video_modified_or_copied': False,
            'frame_cache_bulk_generated': False,
            'latest_bundle_path': str(LATEST_DIR),
            'reference_sources': REFERENCE_SOURCES,
            'package_install_stdout_tail': install_info.get('install', {}).get('stdout', '')[-4000:],
            'package_install_stderr_tail': install_info.get('install', {}).get('stderr', '')[-4000:],
            'actual_runtime_seconds': round(time.time() - start, 3),
        }
        SUMMARY_MD.write_text(make_summary(report), encoding='utf-8')
        write_json(REPORT_JSON, report)
        write_log()

        log('[STEP 10] Run Sub Agent validations')
        report['self_validation'] = {
            'environment_safety': 'PASS' if not protected_modified and env['torch_version'] == '2.11.0+cu128' else 'WARN',
            'download_validation': 'PASS' if package['available'] and weight['available'] and weight['sha256'] else 'FAIL',
            'smoke_test_validation': 'PASS' if setup_ready else 'FAIL',
            'output_safety': 'PENDING_UNTIL_LATEST_BUNDLE',
        }
        write_json(REPORT_JSON, report)

        log('[STEP 11] Update latest bundle')
        latest = latest_bundle([SCRIPT_PATH, INVENTORY_CSV, SUMMARY_MD, REPORT_JSON, RUN_LOG], warnings)
        report['latest_bundle_path'] = latest['path']
        report['latest_bundle_files'] = latest['files']
        report['latest_bundle_forbidden_files'] = latest['forbidden_files']
        report['self_validation']['output_safety'] = 'PASS' if not latest['forbidden_files'] else 'FAIL'
        write_json(REPORT_JSON, report)
        SUMMARY_MD.write_text(make_summary(report), encoding='utf-8')
        shutil.copy2(REPORT_JSON, LATEST_DIR / REPORT_JSON.name)
        shutil.copy2(SUMMARY_MD, LATEST_DIR / SUMMARY_MD.name)

        log('[STEP 12] Print final summary')
        log(f"setup_status={report['setup_status']}")
        log(f"install_performed={report['install_performed']} download_performed={report['download_performed']}")
        log(f"package={report['transnetv2_package_name']} version={report['transnetv2_package_version']}")
        log(f"weight_path={report['weight_path']} sha256={report['weight_sha256']}")
        log(f"smoke_test_status={report['smoke_test_status']} ready={report['ready_for_next_extraction_step']}")
        log(f"warnings={json.dumps(warnings, ensure_ascii=False)}")
        log(f"errors={json.dumps(errors, ensure_ascii=False)}")
        write_log()
        shutil.copy2(RUN_LOG, LATEST_DIR / RUN_LOG.name)
        return 0 if setup_ready else 1
    except Exception as exc:
        errors.append({'exception': repr(exc)})
        log(f'[ERROR] {repr(exc)}')
        write_json(REPORT_JSON, {'task_name': TASK_NAME, 'project_root': str(PROJECT_ROOT), 'setup_status': 'ERROR', 'warnings': warnings, 'errors': errors})
        write_log()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
