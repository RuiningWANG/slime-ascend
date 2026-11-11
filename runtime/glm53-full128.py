#!/usr/bin/env python3
"""Full-128 topology/check/lifecycle helpers. Static modes use only the stdlib."""
import argparse
import glob
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time


def topology(path):
    cfg = json.loads(Path(path).read_text())
    nodes = cfg['nodes']
    assert cfg['world_size'] == 128 and cfg['group_size'] in (8, 16)
    assert len(nodes) in (8, 12) and len({n['ip'] for n in nodes}) == len(nodes)
    assert nodes == sorted(nodes, key=lambda n: ipaddress.ip_address(n['ip']))
    groups = {}
    for n in nodes:
        assert len(n['devices']) in (8, 16) and len(n['devices']) % cfg['group_size'] == 0
        assert n['devices'] == list(range(min(n['devices']), max(n['devices']) + 1))
        groups.setdefault(n['physical_host'], []).extend(n['devices'])
    assert len(groups) == 8 and all(sorted(v) == list(range(16)) for v in groups.values())
    assert sum(len(n['devices']) for n in nodes) == 128
    return cfg


def local_node(cfg):
    import fcntl
    found = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for n in cfg['nodes']:
            try:
                raw = fcntl.ioctl(s, 0x8915, struct.pack('256s', n['interface'].encode()[:15]))
                if socket.inet_ntoa(raw[20:24]) == n['ip']:
                    found.append(n)
            except OSError:
                pass
    assert len(found) == 1, f'Expected one configured local IP, found {found}'
    n = found[0]
    devices = sorted(int(Path(p).name[7:]) for p in glob.glob('/dev/davinci[0-9]*'))
    assert devices == n['devices'], (n['ip'], 'device allocation changed', devices, n['devices'])
    return n


def option(args, name, default=None):
    if name not in args:
        assert default is not None, f'Missing {name}'
        return default
    assert args.count(name) == 1, f'Duplicate {name}'
    return args[args.index(name) + 1]


def check(cfg, checkpoint, data, args, assets=True):
    assert int(option(args, '--actor-num-nodes')) * int(option(args, '--actor-num-gpus-per-node')) == 128
    assert int(option(args, '--num-gpus-per-node')) == cfg['group_size']
    assert int(option(args, '--actor-num-gpus-per-node')) == cfg['group_size']
    assert int(option(args, '--actor-num-nodes')) == 128 // cfg['group_size']
    assert int(option(args, '--rollout-num-gpus')) == 128
    tp, pp, cp, ep = [int(option(args, '--' + k)) for k in (
        'tensor-model-parallel-size', 'pipeline-model-parallel-size',
        'context-parallel-size', 'expert-model-parallel-size')]
    assert tp == 1 and pp == 4 and ep in (16, 32)
    assert 128 % (tp*pp*cp) == 0 and 128 % (tp*pp*ep) == 0 and 288 % ep == 0
    assert int(option(args, '--num-experts')) == 288 and int(option(args, '--num-layers')) == 45
    batch = int(option(args, '--global-batch-size'))
    dp = 128 // (tp*pp*cp)
    assert batch > 0 and batch % dp == 0
    assert batch == int(option(args, '--rollout-batch-size')) * int(option(args, '--n-samples-per-prompt'))
    context = int(option(args, '--seq-length'))
    assert context > int(option(args, '--rollout-max-response-len')) > 0
    assert int(option(args, '--num-rollout')) > 0
    assert int(option(args, '--rollout-num-gpus-per-engine')) == 64
    assert '--sglang-language-only' in args and '--use-rollout-routing-replay' in args
    assert '--colocate' in args and option(args, '--rollout-data-transport') == 'object-store'
    assert int(option(args, '--decoder-first-pipeline-num-layers')) == 12
    assert int(option(args, '--decoder-last-pipeline-num-layers')) == 11
    if not assets:
        print(json.dumps(dict(check_config='PASS', containers=len(cfg['nodes']),
            world=128, group_size=cfg['group_size'], tp=tp, pp=pp, cp=cp, ep=ep, dp=dp, batch=batch)))
        return
    root = Path(checkpoint)
    assert (root/'config.json').is_file(), f'Missing checkpoint config: {root}; set HF_CHECKPOINT to the complete full288 BF16 directory'
    assert (root/'model.safetensors.index.json').is_file(), f'Missing checkpoint index: {root}; set HF_CHECKPOINT to the complete full288 BF16 directory'
    model = json.loads((root/'config.json').read_text())
    model = model.get('text_config', model)
    expected = dict(num_hidden_layers=45, hidden_size=4096, n_routed_experts=288,
                    num_experts_per_tok=8, vocab_size=154880)
    for key, value in expected.items():
        assert model[key] == value, (key, model.get(key), value)
    index = json.loads((root/'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    assert shards and all((root/name).is_file() for name in shards), 'Checkpoint shard missing'
    p = Path(data)
    assert p.is_file() and p.stat().st_size > 0, 'Prompt data missing/empty'
    if p.suffix == '.parquet':
        with p.open('rb') as f:
            assert f.read(4) == b'PAR1'
    else:
        with p.open() as f:
            row = next(json.loads(line) for line in f if line.strip())
            assert 'prompt' in row and 'label' in row
    print(json.dumps(dict(check='PASS', containers=len(cfg['nodes']), physical_hosts=8,
        world=128, logical_groups=128 // cfg['group_size'], dies_per_group=cfg['group_size'], tp=tp, pp=pp, cp=cp, ep=ep,
        dp=dp, batch=batch, experts=288, layers=45, checkpoint_shards=len(shards))))


def idle(node, destination, ignore_warning=False):
    p = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
    Path(destination).write_text(p.stdout+p.stderr)
    assert p.returncode == 0, 'npu-smi failed; see npu-before.txt'
    rows = [line for line in p.stdout.splitlines() if re.search(r'\b[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]\b', line)]
    assert len(rows) == len(node['devices']), ('Unexpected die count', len(rows))
    utilization = [re.search(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]\s*\|\s*(\d+)', row) for row in rows]
    assert all(m and int(m[1]) == 0 for m in utilization), 'AICore is active or snapshot format is unknown'
    blocked = 'Alarm|Critical|UNKNOWN' if ignore_warning else 'Warning|Alarm|Critical|UNKNOWN'
    assert not re.search(r'\|\s*(?:' + blocked + r')\s*\|', p.stdout, re.I), 'NPU health blocks launch'
    section = p.stdout.split('Process id', 1)
    assert len(section) == 2 and section[1].count('No running processes found') == len(node['devices'])//2, 'NPU processes exist or snapshot is incomplete'


def register(cfg, node, root, args):
    root = Path(root)
    contract = dict(topology=cfg, args=args)
    p = root/'contract.json'
    content = json.dumps(contract, sort_keys=True)
    try:
        with p.open('x') as f:
            f.write(content)
    except FileExistsError:
        # Another container may still be writing this small NFS contract.
        for _ in range(50):
            try:
                existing = p.read_text()
                if existing == content:
                    break
            except OSError:
                pass
            time.sleep(.1)
        else:
            raise RuntimeError('Run ID already has different topology/arguments')
    report = dict(ip=node['ip'], physical_host=node['physical_host'], devices=node['devices'],
                  runtime_devices=list(range(len(node['devices']))),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  argument_sha256=hashlib.sha256(content.encode()).hexdigest())
    (root/f"node-{node['ip']}"/'identity.json').write_text(json.dumps(report, indent=2))


def ready(cfg, address, root):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    root = Path(root)
    ray.init(address=address)
    expected = {n['ip']: len(n['devices']) for n in cfg['nodes']}
    deadline = time.monotonic()+900
    while time.monotonic() < deadline:
        assert not list(root.glob('failed-*.json')), 'Peer launcher failed'
        found = {n['NodeManagerAddress']: n['Resources'].get('NPU',0) for n in ray.nodes() if n['Alive']}
        assert not (found.keys()-expected.keys()), ('Unexpected Ray nodes', found)
        if found == expected:
            break
        print('Waiting for selected containers:', found, flush=True)
        time.sleep(5)
    else:
        raise TimeoutError(f'Not all {len(cfg["nodes"])} containers / 128 dies registered')
    identities = [json.loads((root/f"node-{n['ip']}"/'identity.json').read_text()) for n in cfg['nodes']]
    by_host = {}
    for n in identities:
        by_host.setdefault(n['physical_host'], set()).add(n['boot_id'])
    assert all(len(v)==1 for v in by_host.values()) and len(set.union(*by_host.values()))==8, 'Host pairing changed'
    pg = placement_group([{'NPU':1,'CPU':1} for _ in range(128)], strategy='PACK')
    actors = []
    try:
        ray.get(pg.ready(), timeout=120)
        @ray.remote(num_cpus=.1, resources={'NPU':1})
        class Slot:
            def identity(self):
                return ray.util.get_node_ip_address(), ray.get_runtime_context().get_accelerator_ids()['NPU'][0]
        actors = [Slot.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(128)]
        slots = sorted(ray.get([a.identity.remote() for a in actors], timeout=120),
                       key=lambda x:(ipaddress.ip_address(x[0]),int(x[1])))
        # ACL renumbers an upper-half container's physical 8..15 to local 0..7.
        # Ray/torch/SGLang use these local IDs; /dev and npu-smi use physical IDs.
        expect = [(n['ip'],str(d)) for n in cfg['nodes'] for d in range(len(n['devices']))]
        assert [(ip,str(d)) for ip,d in slots] == expect, ('Ray device ID map differs',slots)
        group_size = cfg['group_size']
        for start in range(0,128,group_size):
            chunk=slots[start:start+group_size]
            assert len({ip for ip,_ in chunk})==1 and [int(d) for _,d in chunk]==list(range(int(chunk[0][1]),int(chunk[0][1])+group_size))
        physical={n['ip']:n['devices'] for n in cfg['nodes']}
        (root/'rank-map.json').write_text(json.dumps([
            dict(rank=i,ip=ip,runtime_device=int(d),physical_device=physical[ip][int(d)],
                 logical_group=i//group_size,engine=i//64,engine_node=(i%64)//group_size)
            for i,(ip,d) in enumerate(slots)],indent=2))
        print(f'PLACEMENT PASS: 128 unique dies; {128 // group_size} local groups; two TP64 engines',flush=True)
    finally:
        for actor in actors:
            ray.kill(actor,no_restart=True)
        remove_placement_group(pg)
        ray.shutdown()


def cleanup(token):
    marker=('GLM53_FULL128_RUN_ID='+token).encode()
    protected={os.getpid()}
    pid=os.getppid()
    while pid>1:
        protected.add(pid)
        try: pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError): break
    def scan():
        result={}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) in protected: continue
            try:
                if marker in (p/'environ').read_bytes().split(b'\0'):
                    st=(p/'stat').read_text().rsplit(')',1)[1].split()
                    if st[0]!='Z': result[int(p.name)]=st[19]
            except (OSError,IndexError): pass
        return result
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,ticks in scan().items():
            try:
                now=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                if now[19]==ticks: os.kill(pid,sig)
            except (OSError,IndexError): pass
        time.sleep(3)
    left=scan()
    print(json.dumps({'cleanup_run_id':token,'remaining_owned_processes':left}))
    assert not left, 'Some task-owned processes remain'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','check-config','node','idle','register','ready','cleanup','native'])
    parser.add_argument('--topology',required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--data')
    parser.add_argument('--root')
    parser.add_argument('--address')
    parser.add_argument('--token')
    # Keep framework CLI tokens after '--' separate from helper options.
    raw=sys.argv[1:]
    i=raw.index('--') if '--' in raw else len(raw)
    ns=parser.parse_args(raw[:i]); ns.args=raw[i+1:] if i<len(raw) else []
    cfg=topology(ns.topology)
    if ns.action in ('check','check-config'): check(cfg,ns.checkpoint,ns.data,ns.args,ns.action=='check')
    elif ns.action=='node':
        n=local_node(cfg)
        for key,val in dict(NODE_IP=n['ip'],NODE_DIES=len(n['devices']),COMM_IFNAME=n['interface'],
                            PHYSICAL_DIES=','.join(map(str,n['devices'])),
                            VISIBLE_DIES=','.join(map(str,range(len(n['devices'])))),
                            HEAD_IP=cfg['nodes'][0]['ip']).items():
            print('export '+key+'='+shlex.quote(str(val)))
    elif ns.action=='idle': idle(local_node(cfg),ns.root,cfg.get('ignore_npu_warning',False))
    elif ns.action=='register': register(cfg,local_node(cfg),ns.root,ns.args)
    elif ns.action=='ready': ready(cfg,ns.address,ns.root)
    elif ns.action=='cleanup': cleanup(ns.token)
    elif ns.action=='native':
        import torch
        from slime.utils import accelerator
        from slime.utils.arguments import parse_args
        sys.argv=['full128-native-check']+ns.args
        a=parse_args()
        assert a.actor_num_nodes*a.actor_num_gpus_per_node==128
        assert a.num_experts==288 and a.num_layers==45 and a.sglang_tp_size==64
        assert a.offload_train and a.offload_rollout
        ids=os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
        assert torch.npu.device_count()==len(ids), 'Visible NPU count differs from allocation'
        assert [accelerator.resolve_visible_device_id(d) for d in ids]==list(range(len(ids)))
        assert not torch.npu.is_initialized(), 'Parser initialized the NPU unexpectedly'
        print('NATIVE ARGUMENTS PASS; visible='+','.join(ids)+'; NPU not initialized',flush=True)


if __name__=='__main__':
    main()
