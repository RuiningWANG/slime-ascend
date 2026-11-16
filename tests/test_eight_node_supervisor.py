import importlib.util
import os
from pathlib import Path
import unittest

source = Path(os.environ.get('EIGHT_NODE_SUPERVISOR_SOURCE', str(Path(__file__).resolve().parents[1] / 'runtime/lh-eight-node-supervisor.py')))
spec = importlib.util.spec_from_file_location('eight_node_supervisor', source)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

class ComputeProcessMatchingTests(unittest.TestCase):
    markers = ('site-packages/ray/', 'ray::', 'verl.trainer', 'sglang.launch_server', 'run-glm5.3-flash', 'torchrun')

    def test_real_compute_entrypoints(self):
        cases = [
            ['/opt/ray/raylet'],
            ['/opt/ray/gcs_server'],
            ['ray::WorkerDict.actor_rollout_init_model'],
            ['python3', '-m', 'verl.trainer.main_ppo'],
            ['python3', '-u', '-m', 'sglang.launch_server'],
            ['bash', '/project/scripts/run-glm5.3-flash-300B-8nodes-npu.sh'],
            ['/env/bin/ray', 'start', '--block'],
            ['python3', '/env/bin/ray', 'start', '--block'],
            ['/env/bin/vllm', 'serve', '/model'],
            ['/env/bin/torchrun', '--nproc-per-node', '8', 'train.py'],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertTrue(guard.is_compute_process({'argv': argv}, self.markers))

    def test_readonly_probe_source_is_not_a_compute_entrypoint(self):
        code = 'names = ["ray::", "torchrun", "verl.trainer", "site-packages/ray/"]; print(names)'
        cases = [
            ['python3', '-c', code],
            ['bash', '-c', 'python3 -c ' + repr(code)],
            ['python3', '/tools/readonly_probe.py', 'ray::WorkerDict'],
            ['npu-smi', 'info'],
            ['opencode', 'serve', '--port', '4097'],
            ['ray', 'status'],
            ['python3', '/env/bin/ray', 'status'],
            ['vllm', '--help'],
            [],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertFalse(guard.is_compute_process({'argv': argv}, self.markers))

    def test_identity_retains_structured_arguments(self):
        row = guard.identity(os.getpid())
        self.assertTrue(row['argv'])
        self.assertEqual(row['cmd'], ' '.join(row['argv']))
        self.assertTrue(guard.alive(row))

    def test_reused_pid_identity_is_not_signalled(self):
        row = guard.identity(os.getpid())
        wrong = dict(row, ticks=str(int(row['ticks']) + 1))
        self.assertFalse(guard.alive(wrong))
        self.assertFalse(guard.signal_identity(wrong, 15))

if __name__ == '__main__':
    unittest.main()
