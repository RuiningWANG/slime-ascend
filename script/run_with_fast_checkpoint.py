#!/usr/bin/python3
"""Mount only this job's checkpoint directory on the head's local disk.

The model recipe, checkpoint serialization, optimizer and numerical code are
unchanged. All mounts are private to this launch and its descendants.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


WORK = Path(__file__).resolve().parents[1]
LOGICAL = WORK / 'checkpoints/long150_sharedtp1'
BACKEND = Path(os.environ['GLM53_FASTIO_BACKEND'])
FAST_IPS = os.environ['GLM53_NODE_IPS'].split()
DEADLINE = datetime.datetime.fromisoformat(os.environ['GLM53_FASTIO_DEADLINE']).timestamp()
TOOLS = Path(os.environ['GLM53_FASTIO_TOOL_CACHE']) / 'extracted'


def ensure_tools(clean):
    """Extract pinned, signed RPMs locally without changing host packages."""
    vendor = WORK / 'dependencies/fastio'
    data = (vendor / 'tool-manifest.json').read_bytes()
    key = hashlib.sha256(data).hexdigest()
    cache = TOOLS.parent
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (cache / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = cache / '.prepared-key'
        if ready.exists() and ready.read_text().strip() == key and (TOOLS / 'usr/bin/sshfs').is_file():
            return
        TOOLS.mkdir(exist_ok=True)
        for item in json.loads(data)['packages']:
            name = item['url'].rsplit('/', 1)[1]
            source = vendor / 'packages' / name
            payload = source.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == item['sha256'], f'Package changed: {name}'
            package = cache / name
            package.write_bytes(payload)
            subprocess.run(['rpm', '-K', str(package)], env=clean, check=True, capture_output=True)
            unpack = subprocess.Popen(['rpm2cpio', str(package)], env=clean, stdout=subprocess.PIPE)
            try:
                subprocess.run(['cpio', '-idm', '--quiet', '--no-absolute-filenames'],
                               stdin=unpack.stdout, cwd=TOOLS, env=clean, check=True,
                               capture_output=True, timeout=20)
            finally:
                unpack.stdout.close()
            assert unpack.wait(timeout=5) == 0
        for path in TOOLS.rglob('*'):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & 0o777)
        ready.write_text(key + '\n')


def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def own_ray_groups(marker, known):
    groups = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            token = identity(pid)
            if token is None:
                continue
            owned = known.get(pid) == token
            if not owned:
                env = dict(x.split(b'=', 1) for x in (path / 'environ').read_bytes().split(b'\0') if b'=' in x)
                owned = env.get(b'TMPDIR') == marker or env.get(b'RAY_TMPDIR') == marker + b'/ray'
            if owned:
                known[pid] = token
                groups.add(os.getpgid(pid))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return groups


def signal_owned(groups, known, sig):
    """Signal only identities observed in this run, including detached actors."""
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            if os.getpgid(pid) in groups and known.get(pid) == identity(pid):
                os.kill(pid, sig)
        except (FileNotFoundError, ProcessLookupError):
            pass


def signal_child_group(child, sig):
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true', help='CPU storage check only; no model or Ray')
    parser.add_argument('--inner', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--outer-mnt-ns', help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_id)
    assert time.time() < DEADLINE, 'Allocated resource window has ended'
    if not args.inner:
        outer = os.readlink('/proc/self/ns/mnt')
        command = ['/usr/bin/unshare', '--mount', '--propagation', 'private',
                   '/usr/bin/python3', '-I', '-S', str(Path(__file__).resolve()),
                   '--inner', '--outer-mnt-ns', outer, *sys.argv[1:]]
        os.execv(command[0], command)
    assert re.fullmatch(r'mnt:\[[0-9]+\]', args.outer_mnt_ns or ''), 'Missing outer namespace identity'
    current_namespace = os.readlink('/proc/self/ns/mnt')
    assert current_namespace != args.outer_mnt_ns, 'Private mount namespace required'
    assert current_namespace != os.readlink('/proc/1/ns/mnt'), 'Refusing to mount in the node init namespace'
    clean = dict(os.environ)
    clean.pop('LD_PRELOAD', None)
    clean.pop('LD_LIBRARY_PATH', None)
    clean['PATH'] = '/usr/bin:/usr/sbin:/bin:/sbin:' + os.environ.get('PATH', '')
    assert shutil.which('npu-smi', path=clean['PATH']), 'Ascend driver tools missing from the inherited PATH'
    interface = clean.get('FULL_COMM_IFNAME', 'enp39s0f0')
    addresses = subprocess.check_output(['ip', '-o', '-4', 'addr', 'show', 'dev', interface], env=clean, text=True)
    found = set(re.findall(r'\binet ([0-9.]+)/', addresses)) & set(FAST_IPS)
    assert len(found) == 1, f'Expected one configured interface address: {found}'
    node = 21 + FAST_IPS.index(found.pop())
    context = Path('/tmp/glm53-fastio-context') / args.run_id / f'node{node}'
    context.mkdir(mode=0o700, parents=True, exist_ok=False)
    assert LOGICAL.is_dir(), f'Checkpoint mountpoint must already exist: {LOGICAL}'
    manifest = {'run_id': args.run_id, 'node': node, 'logical_checkpoint_root': str(LOGICAL),
                'physical_head_checkpoint_root': str(BACKEND), 'head_fast_ip': FAST_IPS[0],
                'mount_namespace': os.readlink('/proc/self/ns/mnt'), 'npu_used_by_wrapper': False,
                'native_checkpoint_format_unchanged': True, 'deadline': DEADLINE}
    (context / 'location.json').write_text(json.dumps(manifest, indent=2))
    fs = child = None
    fs_log = None
    known = {}
    marker = ('/tmp/g53e-0/' + hashlib.sha256(args.run_id.encode()).hexdigest()[:12]).encode()
    exit_code = 1
    mounted = False

    def interrupted(signum, frame):
        nonlocal exit_code
        exit_code = 128 + signum
        raise SystemExit(exit_code)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if node == 21:
            assert '/data' in subprocess.check_output(['findmnt', '-n', '-T', str(BACKEND.parent.parent), '-o', 'TARGET'], env=clean, text=True).split()
            BACKEND.mkdir(mode=0o700, parents=True, exist_ok=True)
            free = shutil.disk_usage(BACKEND).free
            assert free >= 4 * 1024**4, f'Insufficient free head disk for a full checkpoint: {free}'
            subprocess.run(['mount', '--bind', str(BACKEND), str(LOGICAL)], env=clean, check=True)
        else:
            ensure_tools(clean)
            password_file = Path(clean.pop('GLM53_FASTIO_PASSWORD_FILE'))
            secret_dir = Path('/tmp/glm53-fastio-auth')
            assert password_file.parent == secret_dir and password_file.is_file() and not password_file.is_symlink()
            st = password_file.stat()
            assert st.st_uid == os.getuid() and st.st_mode & 0o077 == 0
            host_key = context / 'known_hosts'
            shutil.copyfile(Path(clean.pop('GLM53_FASTIO_KNOWN_HOSTS')), host_key)
            host_key.chmod(0o600)
            tool_env = dict(clean, LD_LIBRARY_PATH=str(TOOLS / 'usr/lib64'))
            tool_env['PATH'] = str(TOOLS / 'usr/bin') + ':' + clean['PATH']
            options = ('password_stdin,StrictHostKeyChecking=yes,UserKnownHostsFile=' + str(host_key) +
                       ',ServerAliveInterval=10,ServerAliveCountMax=2,ConnectTimeout=5,reconnect,'
                       'cache=no,attr_timeout=0,entry_timeout=0,negative_timeout=0,direct_io')
            fs_log = (context / 'sshfs.log').open('wb')
            with password_file.open('rb') as password:
                fs = subprocess.Popen([str(TOOLS / 'usr/bin/sshfs'), '-f',
                     f'root@{FAST_IPS[0]}:{BACKEND}', str(LOGICAL), '-o', options],
                     stdin=password, stdout=fs_log, stderr=fs_log, env=tool_env, start_new_session=True)
            stop = min(time.monotonic() + 30, time.monotonic() + DEADLINE - time.time())
            while not os.path.ismount(LOGICAL):
                assert fs.poll() is None, 'SSHFS startup failed; see local context sshfs.log'
                assert time.monotonic() < stop, 'SSHFS startup timed out'
                time.sleep(0.1)
            password_file.unlink()
        mounted = True
        assert time.time() < DEADLINE, 'Allocated resource window ended during storage setup'
        if args.smoke:
            path = LOGICAL / f'.cpu-storage-smoke-node{node}-{os.getpid()}'
            payload = hashlib.shake_256(f'glm53-fastio-{node}'.encode()).digest(1024 * 1024)
            start = time.monotonic()
            try:
                with path.open('wb') as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                assert path.read_bytes() == payload
                manifest['smoke_write_fsync_read_seconds'] = time.monotonic() - start
                manifest['smoke_bytes'] = len(payload)
                manifest['smoke_passed'] = True
            finally:
                path.unlink(missing_ok=True)
            exit_code = 0
        else:
            entry = Path(args.entry).resolve()
            assert entry.parent == WORK / 'script' and entry.is_file()
            child = subprocess.Popen(['/bin/bash', str(entry), '--run-id', args.run_id], env=clean, start_new_session=True)
            next_scan = 0
            while child.poll() is None and time.time() < DEADLINE:
                if time.monotonic() >= next_scan:
                    own_ray_groups(marker, known)
                    next_scan = time.monotonic() + 5
                time.sleep(0.5)
            exit_code = child.returncode if child.returncode is not None else 124
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        groups = own_ray_groups(marker, known)
        signal_child_group(child, signal.SIGTERM)
        signal_owned(groups, known, signal.SIGTERM)
        time.sleep(1)
        signal_owned(groups, known, signal.SIGKILL)
        signal_child_group(child, signal.SIGKILL)
        if mounted or os.path.ismount(LOGICAL):
            try:
                subprocess.run(['umount', str(LOGICAL)], env=clean, check=True, capture_output=True, timeout=3)
            except (subprocess.SubprocessError, OSError):
                try:
                    subprocess.run(['umount', '-l', str(LOGICAL)], env=clean, check=True, capture_output=True, timeout=3)
                except (subprocess.SubprocessError, OSError) as error:
                    manifest['private_unmount_error'] = str(error)
        if fs is not None and fs.poll() is None:
            signal_child_group(fs, signal.SIGTERM)
            try:
                fs.wait(timeout=2)
            except subprocess.TimeoutExpired:
                signal_child_group(fs, signal.SIGKILL)
                fs.wait(timeout=2)
        if fs_log is not None:
            fs_log.close()
        manifest['wrapper_exit_code'] = exit_code
        manifest['checkpoint_mount_removed'] = not os.path.ismount(LOGICAL)
        (context / 'result.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == '__main__':
    sys.exit(main())
