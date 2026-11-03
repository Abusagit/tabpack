#!/usr/bin/env python3
"""Send an immutable snapshot of this Git worktree to a GPU host and run Python.

Example: python3 scripts/gpu_run.py --gpu 4 -- run_profiling_experiment.py configs/paper_california.json --output ../output
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def output(*args, **kwargs):
    return subprocess.check_output(args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='ysda_imladris')
    parser.add_argument('--gpu', required=True, help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--remote-root', default='tabpack-runs', help='Relative path under remote HOME')
    parser.add_argument('--remote-python', default='~/tabpack/.venv/bin/python')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Python script/arguments after --')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide Python arguments after --')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.host):
        parser.error('host must be a simple SSH alias or user@host')
    if not re.fullmatch(r'[0-9,]+', args.gpu):
        parser.error('gpu must be a CUDA device index or comma-separated indices')
    remote_root = Path(args.remote_root)
    if remote_root.is_absolute() or not remote_root.parts or any(p in ('.', '..') for p in remote_root.parts):
        parser.error('remote-root must be a relative path under HOME without ..')
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', args.remote_root):
        parser.error('remote-root contains unsupported characters')
    if not (args.remote_python.startswith('~/') or args.remote_python.startswith('/')):
        parser.error('remote-python must be an absolute path or start with ~/')
    root = Path(output('git', 'rev-parse', '--show-toplevel', text=True).strip()).resolve()
    os.chdir(root)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    local_dir = root / 'artifacts' / 'runs' / run_id
    local_dir.mkdir(parents=True, exist_ok=False)
    remote_dir = args.remote_root.rstrip('/') + '/' + run_id
    files = output('git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0')
    paths = sorted({os.fsdecode(p) for p in files if p})
    manifest = []
    with tempfile.TemporaryDirectory(prefix='tabpack-snapshot-') as temporary:
        snap = Path(temporary) / 'code'
        snap.mkdir()
        for relative in paths:
            source = root / relative
            if not source.exists() and not source.is_symlink():
                continue  # tracked file removed in this worktree
            if source.is_symlink():
                raise RuntimeError('Refusing symlink in snapshot: ' + relative)
            if not source.is_file():
                raise RuntimeError('Unsupported non-file in snapshot: ' + relative)
            destination = snap / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest.append({'path': relative, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'bytes': destination.stat().st_size})
        metadata = {'run_id': run_id, 'worktree': str(root), 'commit': output('git', 'rev-parse', 'HEAD', text=True).strip(),
                    'branch': output('git', 'branch', '--show-current', text=True).strip(), 'host': args.host, 'gpu': args.gpu,
                    'remote_python': args.remote_python, 'python_arguments': command, 'remote_dir': remote_dir,
                    'files': manifest}
        (local_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        (local_dir / 'git-status.txt').write_bytes(output('git', 'status', '--porcelain=v1', '-z'))
        (local_dir / 'git-diff.patch').write_bytes(output('git', 'diff', '--binary', 'HEAD', '--'))
        # Remote paths are unique per run. Never delete or mutate another run.
        run('ssh', args.host, 'mkdir -p ' + shlex.quote(remote_dir + '/code') + ' ' + shlex.quote(remote_dir + '/output') + ' ' + shlex.quote(remote_dir + '/metadata'))
        run('rsync', '-a', str(snap) + '/', args.host + ':' + remote_dir + '/code/')
        run('rsync', '-a', str(local_dir) + '/', args.host + ':' + remote_dir + '/metadata/')
        # Run from snapshot root so imports resolve against snapshot, not server checkout.
        python = ('"$HOME"/' + shlex.quote(args.remote_python[2:])) if args.remote_python.startswith('~/') else shlex.quote(args.remote_python)
        shell = ('set -e; cd ' + shlex.quote(remote_dir + '/code') + '; '
                 'export CUDA_VISIBLE_DEVICES=' + shlex.quote(args.gpu) + '; '
                 'export PYTHONUNBUFFERED=1; export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"; '
                 + python + ' ' + shlex.join(command))
        print('Run:', run_id, '\nLocal:', local_dir, '\nRemote:', args.host + ':' + remote_dir, flush=True)
        environment = ('nvidia-smi; ' + python + ' -c ' + shlex.quote('import sys, torch; print("Python:", sys.version); print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)'))
        env = subprocess.run(['ssh', args.host, environment], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (local_dir / 'environment.txt').write_bytes(env.stdout)
        with (local_dir / 'run.log').open('wb') as log:
            proc = subprocess.Popen(['ssh', args.host, shell], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                code = 130
        # Preserve output even if the experiment fails.
        copy = subprocess.run(['rsync', '-a', args.host + ':' + remote_dir + '/output/', str(local_dir / 'output') + '/'])
        (local_dir / 'exit-code.txt').write_text(str(code) + '\n')
        if copy.returncode:
            print('Could not fetch outputs; they remain on the server:', remote_dir + '/output', file=sys.stderr)
            return copy.returncode
        print('Results:', local_dir, flush=True)
        return code


if __name__ == '__main__':
    sys.exit(main())
