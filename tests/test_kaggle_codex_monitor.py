import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('monitor', Path(__file__).resolve().parents[1] / 'scripts/watch_kaggle_codex.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'jobs': [{'id': 'owner/a'}, {'id': 'owner/b', 'download': 'output'}]}

    def test_unchanged_checks_do_not_spam(self):
        view = monitor.snapshot(self.plan, {}, 10)
        self.assertEqual(monitor.reasons(view, view, 60, 900), [])
        self.assertEqual(monitor.reasons(view, view, 900, 900), ['periodic health check'])

    def test_completed_jobs_require_requested_downloads(self):
        state = {'owner/a': {'status': 'complete'}, 'owner/b': {'status': 'complete'}}
        self.assertFalse(monitor.snapshot(self.plan, state, 0)['complete'])
        state['owner/b']['downloaded'] = True
        view = monitor.snapshot(self.plan, state, 10000)
        self.assertTrue(view['complete'])
        self.assertFalse(view['controller_stale'])

    def test_stale_controller_and_errors_are_events(self):
        before = monitor.snapshot(self.plan, {}, 0)
        after = monitor.snapshot(self.plan, {'owner/a': {'status': 'running', 'last_read_error': 'SECRET'}}, 301)
        events = monitor.reasons(before, after, 60, 900)
        self.assertTrue(any('read error' in e for e in events))
        self.assertTrue(any('five minutes' in e for e in events))
        self.assertNotIn('SECRET', monitor.message(Path('project'), events))

    def test_repeated_error_counts_do_not_trigger_repeated_events(self):
        state = {'owner/a': {'status': 'running', 'last_read_error': 'error', 'read_failures': 1}}
        before = monitor.snapshot(self.plan, state, 0)
        state['owner/a']['read_failures'] = 2
        after = monitor.snapshot(self.plan, state, 0)
        self.assertEqual(monitor.reasons(before, after, 60, 900), [])
