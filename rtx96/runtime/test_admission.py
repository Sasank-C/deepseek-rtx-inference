import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from download_gated import admit


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"DSV41_DEPLOYED_IMAGE_DIGEST": "sha256:" + "a" * 64})
        env.start()
        self.addCleanup(env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = Path(self.temp.name) / 'real-test-log.txt'
        p.write_text('unit fixture, not GPU evidence')
        self.guard = dict(status='armed', instance_id=123, epoch=1000, deadline_epoch=9000)
        self.gate = dict(profile='tp4-single-cpu', instance_id=123, deadline_epoch=9000,
            image_digest='sha256:' + 'a' * 64,
            runtime_base_digest='sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34',
            completed_epoch=1000, evidence_files=[str(p)],
            checks={k:'PASS' for k in ['hardware','local_tp4_nccl','kernel','launch_parser',
                                      'external_cleanup_controller','cpu_engram_pinned_memory']})

    def test_parent_cannot_identify_derived_image(self):
        self.gate['image_digest'] = self.gate['runtime_base_digest']
        with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '123')

    def test_other_derived_digest_rejected(self):
        with self.assertRaises(ValueError):
            admit(self.gate, self.guard, 1001, '123', 'sha256:' + 'b' * 64)

    def test_missing_actual_digest_rejected(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '123')

    def test_valid_fixture(self):
        admit(self.gate, self.guard, 1001, '123')

    def test_missing_cpu_gate(self):
        self.gate['checks'].pop('cpu_engram_pinned_memory')
        with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '123')

    def test_split_cannot_reuse_single_host_gate(self):
        self.gate['profile'] = 'tp4-pp2-gpu'
        with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '123')

    def test_old_rental_rejected(self):
        with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '999')

    def test_stale_or_future_guard(self):
        for now in (999, 1020):
            with self.assertRaises(ValueError): admit(self.gate, self.guard, now, '123')

    def test_deadline_reset_rejected(self):
        self.gate['deadline_epoch'] += 3600
        with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '123')

    def test_deleted_evidence_rejected(self):
        Path(self.gate['evidence_files'][0]).unlink()
        with self.assertRaises(ValueError): admit(self.gate, self.guard, 1001, '123')


if __name__ == '__main__': unittest.main()
