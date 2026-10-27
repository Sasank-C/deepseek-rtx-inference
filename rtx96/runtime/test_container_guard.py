"""Pure/mocked tests only: no Vast calls, subprocesses, rentals or GPU work."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget_guard as guard
import arm_container_guard as arm
import preflight_collectives as collective
import preflight_network as network


class GuardTests(unittest.TestCase):
    def arguments(self, path):
        return ['--instance', '101', '--machine', '202', '--interface', 'eth1',
                '--started', '1000', '--deadline', '8000', '--hourly', '5.7',
                '--ingress', '.003', '--egress', '.004', '--node-cap', '14.73',
                '--state', str(path)]

    def check(self, args):
        with patch.object(sys, 'argv', ['budget_guard.py', 'check', *args]), \
             patch.dict(os.environ, {'CONTAINER_ID': '101', 'CONTAINER_API_KEY': 'fixture-scoped'}, clear=True), \
             patch.object(guard, 'reject_saved_cli_credentials'), \
             patch.object(guard.time, 'time', return_value=2000), \
             patch.object(guard.shutil, 'which', return_value='fake'), \
             patch.object(guard, 'supports_noninteractive_destroy', return_value=True), \
             patch.object(guard, 'self_info', return_value={}), \
             patch.object(guard, 'counters', return_value=(0, 0)), \
             patch.object(sys, 'stdout', new_callable=io.StringIO), \
             patch.object(sys, 'stderr', new_callable=io.StringIO):
            guard.main()

    def test_scoped_identity_requires_both_injected_values_and_exact_id(self):
        good = {'CONTAINER_ID': '101', 'CONTAINER_API_KEY': 'fixture-scoped'}
        guard.require_scoped_identity(101, good)
        for environment in ({}, {'VAST_API_KEY': 'fixture-account'},
                            {'CONTAINER_ID': '101'},
                            {'CONTAINER_API_KEY': 'fixture-scoped'},
                            dict(good, CONTAINER_ID='102'),
                            dict(good, CONTAINER_ID='0'),
                            dict(good, CONTAINER_API_KEY=' ')):
            with self.subTest(keys=sorted(environment)):
                with self.assertRaises(ValueError):
                    guard.require_scoped_identity(101, environment)

    def test_missing_scoped_auth_refuses_check_and_watch_before_cli(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            args = self.arguments(Path(folder) / 'state.json')
            for mode in ('check', 'watch'):
                with patch.object(sys, 'argv', ['budget_guard.py', mode, *args]), \
                     patch.dict(os.environ, {'VAST_API_KEY': 'fixture-account'}, clear=True), \
                     patch.object(guard.time, 'time', return_value=2000), \
                     patch.object(guard.subprocess, 'run') as run, \
                     patch.object(sys, 'stderr', new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):
                        guard.main()
                    run.assert_not_called()
            self.assertFalse((Path(folder) / 'state.json').exists())

    def test_cli_environment_overrides_account_key_in_child_only(self):
        original = {'CONTAINER_ID': '101', 'CONTAINER_API_KEY': 'fixture-scoped',
                    'VAST_API_KEY': 'fixture-account', 'PATH': '/fixture/bin'}
        with patch.dict(os.environ, original, clear=True), \
             patch.object(guard, 'reject_saved_cli_credentials'):
            child = guard.cli_environment()
            self.assertEqual(child['VAST_API_KEY'], original['CONTAINER_API_KEY'])
            self.assertEqual(child['VASTAI_NO_UPDATE_CHECK'], '1')
            self.assertEqual(os.environ['VAST_API_KEY'], original['VAST_API_KEY'])

    def test_saved_credentials_refused_without_reading_or_deleting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            candidates = [root / '.vast_api_key',
                          root / '.config/vastai/vast_api_key',
                          root / '.config/vastai/vast_tfa_key',
                          root / 'xdg/vastai/vast_api_key',
                          root / 'xdg/vastai/vast_tfa_key']
            environment = {'XDG_CONFIG_HOME': str(root / 'xdg')}
            with patch.object(guard.Path, 'home', return_value=root):
                guard.reject_saved_cli_credentials(environment)
                for path in candidates:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('public test fixture')
                    with patch.object(guard.Path, 'read_text', side_effect=AssertionError('must not read')):
                        with self.assertRaises(ValueError):
                            guard.reject_saved_cli_credentials(environment)
                    self.assertTrue(path.exists())
                    path.unlink()

    def test_cli_missing_scoped_key_never_spawns_even_with_account_key(self):
        with patch.dict(os.environ, {'CONTAINER_ID': '101', 'VAST_API_KEY': 'fixture-account'}, clear=True), \
             patch.object(guard.subprocess, 'run') as run:
            self.assertIsNone(guard.cli('fake', ['show', 'instance', '101', '--raw'], True))
            self.assertFalse(guard.supports_noninteractive_destroy('fake'))
            run.assert_not_called()

    def test_cap_and_immutable_original_bounds(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / 'state.json'
            args = self.arguments(path)
            self.check(args)
            original = path.read_text()
            self.assertEqual(json.loads(original), arm.pinned_arguments(args))
            for option, value in [('--deadline', '8100'), ('--started', '1100'),
                                  ('--hourly', '5.6'), ('--instance', '102')]:
                changed = args.copy(); changed[changed.index(option) + 1] = value
                with self.assertRaises(SystemExit):
                    self.check(changed)
                self.assertEqual(path.read_text(), original)

    def test_per_node_cap_cannot_exceed_template_ceiling(self):
        self.assertEqual(guard.MAX_NODE_CAP, 14.73)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            args = self.arguments(Path(folder) / 'state.json')
            args[args.index('--node-cap') + 1] = '14.74'
            with self.assertRaises(SystemExit):
                self.check(args)

    def test_expired_original_deadline_refuses_arm(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            args = self.arguments(Path(folder) / 'state.json')
            args[args.index('--deadline') + 1] = '1999'
            with self.assertRaises(SystemExit):
                self.check(args)

    def test_supervisor_refuses_mismatched_pinned_target_before_spawn(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / 'state.json'; args = self.arguments(path)
            state = arm.pinned_arguments(args); state['instance'] = 999
            path.write_text(json.dumps(state))
            with patch.object(arm.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    arm.supervise(args, path, 'fake', 101)
                run.assert_not_called()

    def test_supervisor_watch_exit_triggers_self_cleanup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / 'state.json'; args = self.arguments(path)
            path.write_text(json.dumps(arm.pinned_arguments(args)))
            with patch.object(arm.subprocess, 'run', return_value=subprocess.CompletedProcess([], 2)), \
                 patch.object(guard, 'cleanup') as cleanup:
                arm.supervise(args, path, 'fake', 101)
            cleanup.assert_called_once_with('fake', 101, path.with_name('status.json'),
                                            'watchdog_process_exited')

    def test_cleanup_uses_yes_and_only_own_id(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, \
             patch.object(guard, 'cli') as cli, \
             patch.object(guard.time, 'sleep', side_effect=RuntimeError('bounded test')):
            with self.assertRaises(RuntimeError):
                guard.cleanup('fake', 101, Path(folder) / 'status.json', 'deadline')
            cli.assert_called_once_with('fake', ['destroy', 'instance', '101', '--yes'])

    def test_fresh_heartbeat_requires_identity_and_deadline(self):
        expected = {'instance_id': 101, 'deadline_epoch': 8000}
        good = dict(expected, status='armed', epoch=2001)
        self.assertTrue(arm.confirms_armed(good, expected, 2000, 2002))
        for change in [{'epoch': 1999}, {'epoch': 2010}, {'instance_id': 102},
                       {'deadline_epoch': 8100}, {'status': 'cleanup_requested_unverified'}]:
            self.assertFalse(arm.confirms_armed(dict(good, **change), expected, 2000, 2002))

    def test_arguments_reject_duplicate_ids(self):
        with self.assertRaises(ValueError):
            arm.get_option(['--instance', '101', '--instance=102'], '--instance')

    def test_network_reset_and_cost_trigger(self):
        s = arm.pinned_arguments(self.arguments(Path('/tmp/unused')))
        self.assertEqual(guard.reason_at(2000, s, 0, 0, 1, 0), 'network_counter_reset')
        s['hourly'] = 7.0
        self.assertEqual(guard.reason_at(7999, s, 700_000_000_000, 25_000_000_000, 0, 0),
                         'estimated_cost_ceiling')

    def test_collective_layouts(self):
        single = collective.topology('tp4-single-cpu', 3, 4, 4)
        self.assertIsNone(single['pp_peer'])
        pair = collective.topology('tp4-pp2-gpu', 5, 8, 4)
        self.assertEqual(pair['pp_peer'], 1)
        self.assertEqual(pair['tp_groups'], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(pair['payload_bytes_per_rank'] * 4 * 2 * 5, 419594240)
        with self.assertRaises(ValueError):
            collective.topology('tp4-pp2-gpu', 1, 2, 1)

    def test_private_route_proof_rejects_unrelated_clusters_and_no_vm(self):
        base = {'machine_ids': [101, 202], 'mode': 'overlay'}
        self.assertFalse(network.proof_valid(dict(base, cluster_ids=[None, None])))
        self.assertFalse(network.proof_valid(dict(base, cluster_ids=[5, 6])))
        self.assertTrue(network.proof_valid(dict(base, cluster_ids=[5, 5])))
        self.assertFalse(network.proof_valid({'machine_ids': [101, 101],
                                             'mode': 'overlay', 'cluster_ids': [5, 5]}))
        self.assertFalse(network.proof_valid({'machine_ids': [101, 202],
                                             'mode': 'wireguard', 'vm_eligible': [False, False]}))


if __name__ == '__main__':
    unittest.main()
