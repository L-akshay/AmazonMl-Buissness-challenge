import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cloud_country_audit import audit_country_routes
from src.cloud_routed_decisions import audit_routed_decisions
from src.cloud_store import read_parquet, write_parquet


class RoutedDecisionTests(unittest.TestCase):
    def test_full_population_baseline_reproduction_and_checkpoint_resume(self):
        spec = importlib.util.spec_from_file_location('country_fixture', Path(__file__).with_name('test_cloud_country_audit.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); module.fixture(root)
            # All four folds for a secondary source batch must share one ordered
            # range; country fixtures originally only exercised per-fold grids.
            for path in (root / 'cache/cloud/scores').rglob('*.parquet'):
                b = read_parquet(path); shard = int(path.stem.split('_')[-1])
                b['tid'] = np.array([(2 << 32) + 1000 + shard * 1000 + int(r) * 2
                                     + i // 2 for i, r in enumerate(b['ri'])], dtype=np.uint64)
                # Baseline and specialist candidates need identical identifiers.
                # The extra third row belongs to India and has a separate ID.
                if shard and len(b['ri']) and np.sum(b['ri'] % 2 == 0) == 2:
                    b['tid'][-1] = (2 << 32) + 1000 + shard * 1000 + int(b['ri'][-1]) * 2 + 1
                write_parquet(path, b)
            with patch('src.cloud_country_audit.check_headroom'):
                prior = audit_country_routes(root, {})
            with patch('src.cloud_routed_decisions.check_headroom'), patch('src.cloud_decisions.check_headroom'):
                report = audit_routed_decisions(root, {})
                self.assertEqual(report['rows'], 20)
                self.assertEqual(report['baseline']['confirmation'], prior['routed']['confirmation'])
                self.assertEqual(report['candidate_oracle']['overall']['true_links'], 12)
                (root / 'cache/cloud/routed_decision_audit/report.json').unlink()
                with patch('src.cloud_routed_decisions.competition', side_effect=AssertionError('recomputed')):
                    self.assertEqual(audit_routed_decisions(root, {}), report)

    def test_missing_country_evidence_is_rejected_without_refitting(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'src.cloud_routed_decisions.audit_country_routes', side_effect=AssertionError('unexpected audit')):
            with self.assertRaisesRegex(ValueError, 'completed country-routing'):
                audit_routed_decisions(Path(tmp), {})
