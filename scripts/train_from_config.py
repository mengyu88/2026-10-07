#!/usr/bin/env python3
"""Launch the backed-up train.py from an explicit, portable JSON profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON profile path, relative to the repository or absolute.')
    parser.add_argument('--model-name', help='Hugging Face model ID or an absolute local backbone path.')
    parser.add_argument('--data-dir', help='Directory with train/dev/test.jsonlines, relative to the repository or absolute.')
    parser.add_argument('--run-dir', help='New output directory, relative to the repository or absolute.')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter to run train.py.')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without training or creating directories.')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Additional train.py arguments after -- override the profile.')
    args = parser.parse_args()

    with project_path(args.config).open(encoding='utf-8') as handle:
        profile = json.load(handle)
    data_dir = project_path(args.data_dir or profile['data_dir'])
    run_dir = project_path(args.run_dir or profile['run_dir'])
    model_name = args.model_name or profile['model_name']
    command = [args.python, '-u', str(PROJECT_ROOT / 'train.py')]
    for flag, value in profile['train_args'].items():
        if value is None or value is False:
            continue
        command.append(flag)
        if value is not True:
            command.append(str(value))
    command.extend([
        '--model_name', model_name,
        '--data_dir', str(data_dir),
        '--epoch_metrics_path', str(run_dir / 'epoch_metrics.csv'),
    ])
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ['--'] else args.extra_args
    command.extend(extra_args)
    run_environment = {
        'SPSR_LOG_DIR': str(run_dir / 'logs'),
        'SPSR_CHECKPOINT_DIR': str(run_dir / 'checkpoints'),
        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'),
    }
    print(json.dumps({'cwd': str(PROJECT_ROOT), 'command': command, 'environment': run_environment},
                     ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    missing = [name for name in ('train.jsonlines', 'dev.jsonlines', 'test.jsonlines')
               if not (data_dir / name).is_file()]
    if missing:
        parser.error(f'Missing files in {data_dir}: {", ".join(missing)}')
    if run_dir.exists():
        parser.error(f'Output directory already exists: {run_dir}; choose a new --run-dir.')
    run_dir.mkdir(parents=True)
    (PROJECT_ROOT / 'caches').mkdir(exist_ok=True)
    (run_dir / 'command.txt').write_text(shlex.join(command) + '\n', encoding='utf-8')
    environment = os.environ.copy()
    environment.update(run_environment)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
