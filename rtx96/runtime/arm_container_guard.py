#!/usr/bin/env python3
"""Arm a detached, supervised self-only Vast watchdog; no systemd required.

Pass explicit budget_guard.py options after `--`. Call only in an authorized
rental. A separate controller must still destroy both rentals on failure/deadline.
No credentials are written to disk or added to argv. No model is launched here.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import budget_guard as guard

HERE = Path(__file__).resolve().parent


def get_option(args, name, default=None):
    values = []
    for i, value in enumerate(args):
        if value == name:
            if i + 1 == len(args):
                raise ValueError(f'Missing value for {name}')
            values.append(args[i + 1])
        elif value.startswith(name + '='):
            values.append(value.split('=', 1)[1])
    if len(values) > 1:
        raise ValueError(f'Duplicate {name}')
    if values:
        return values[0]
    if default is None:
        raise ValueError(f'Required: {name}')
    return default


def confirms_armed(value, expected, launched, now):
    try:
        stamp = float(value['epoch'])
        return (value.get('status') == 'armed'
                and value.get('instance_id') == expected['instance_id']
                and value.get('deadline_epoch') == expected['deadline_epoch']
                and launched <= stamp <= now and now - stamp < 8)
    except (ValueError, TypeError, KeyError):
        return False


def pinned_arguments(args):
    integers = {'instance': None, 'machine': None,
                'rx_limit': '700000000000', 'tx_limit': '25000000000'}
    floats = {'started': None, 'deadline': None, 'hourly': None,
              'ingress': None, 'egress': None, 'node_cap': None}
    result = {key: int(get_option(args, '--' + key.replace('_', '-'), default))
              for key, default in integers.items()}
    result.update({key: float(get_option(args, '--' + key.replace('_', '-'), default))
                   for key, default in floats.items()})
    result['interface'] = get_option(args, '--interface')
    return result


def supervise(args, state_path, executable, instance):
    if json.loads(state_path.read_text()) != pinned_arguments(args):
        raise SystemExit('Supervisor arguments differ from immutable pinned state')
    # Separate lock from the immutable self-only watchdog's own lock.
    with open(str(state_path) + '.supervisor.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Container watchdog supervisor already running')
        # This records only safe identities/status; credentials remain in memory.
        guard.write_status(state_path.with_name('supervisor.json'), {
            'pid': os.getpid(), 'instance_id': instance, 'epoch': time.time(),
            'status': 'supervising'})
        subprocess.run([sys.executable, str(HERE / 'budget_guard.py'),
                        'watch', *args], stdin=subprocess.DEVNULL, check=False)
        # watch normally runs until the rental disappears. Unexpected exit must
        # not silently leave a paid instance running, regardless of exit status.
        guard.cleanup(executable, instance, state_path.with_name('status.json'),
                      'watchdog_process_exited')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('guard_options', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.guard_options
    if args[:1] == ['--']:
        args = args[1:]
    if not args:
        parser.error('Pass explicit watchdog options after --')
    if os.geteuid() != 0:
        parser.error('Run inside the rented container as root')
    os.umask(0o077)
    try:
        instance = int(get_option(args, '--instance'))
        # Require an explicit path, start/deadline, and all-in rates; no stale
        # identifiers, implicit clock resets, or copied quote defaults.
        state_path = Path(get_option(args, '--state'))
        for required in ['--machine', '--interface', '--started', '--deadline',
                         '--hourly', '--ingress', '--egress', '--node-cap']:
            get_option(args, required)
        executable = shutil.which(get_option(args, '--cli', 'vastai'))
        if not executable or not state_path.is_absolute():
            raise ValueError('Require official CLI and an absolute state path')
        if not 0 < float(get_option(args, '--node-cap')) <= guard.MAX_NODE_CAP:
            raise ValueError('Per-node estimate cap must be at most $14.73')
        pinned_arguments(args)  # Reject duplicates, missing and malformed bounds.
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if options.supervise:
        # The parent check pinned these exact bounds. Watch revalidates them;
        # this mode is only an internal child path, not an alternative armer.
        if not state_path.is_file() or state_path.is_symlink():
            parser.error('Missing pinned state; run the armer first')
        supervise(args, state_path, executable, instance)
        return
    state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(str(state_path) + '.arm.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another armer is active')
        marker = state_path.with_name('supervisor.json')
        if marker.exists():
            parser.error('Existing supervisor marker; inspect rather than overwrite')
        checked = subprocess.run(
            [sys.executable, str(HERE / 'budget_guard.py'), 'check', *args],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=45, check=False)
        if checked.returncode:
            parser.error('Self-auth/bounds check failed; controller must clean up')
        expected = json.loads(checked.stdout)
        if expected.get('instance_id') != instance:
            parser.error('Unexpected self-check identity')
        log_path = state_path.with_name('supervisor.log')
        with open(log_path, 'xb') as log:
            launched = time.time()
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--supervise',
                 '--', *args], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True, cwd=HERE)
        status_path = state_path.with_name('status.json')
        for _ in range(30):
            try:
                value = json.loads(status_path.read_text())
                if process.poll() is None and confirms_armed(
                        value, expected, launched, time.time()):
                    print(json.dumps({'guard': 'armed', 'supervisor_pid': process.pid,
                                      'instance_id': instance,
                                      'deadline_epoch': expected['deadline_epoch'],
                                      'node_cap_usd': float(get_option(args, '--node-cap'))}))
                    return
                if value.get('status') == 'cleanup_requested_unverified':
                    break
            except (OSError, ValueError):
                pass
            if process.poll() is not None:
                break
            time.sleep(1)
        # Leave the detached cleanup supervisor alive; never kill our safeguard.
        raise SystemExit('Guard did not confirm armed; no setup/weights. Controller must destroy both rentals.')


if __name__ == '__main__':
    main()
