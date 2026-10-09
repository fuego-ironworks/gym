"""Control fixtures test bounded retry machinery; they are not a model corpus."""
from __future__ import annotations
import argparse
import copy
import difflib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import functorial_c_eval as base
import functorial_c_pilot as pilot


class BoundedRepairTests(unittest.TestCase):
    source = 'int value;\n'
    malformed = '--- a/code.c\n+++ b/code.c\n@@\n-int value;\n+int other;\n*** End of File ***\n'
    valid = ''.join(difflib.unified_diff(source.splitlines(keepends=True),
                         ['int other;\n'], fromfile='a/code.c', tofile='b/code.c'))

    def run_fixture(self, first=None, repaired=None, version='0.40.2', fail=False):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'run'
        self.pair = next(p for p in base.read_manifest()['pairs'] if p['id'] == 'fourier-horner')
        self.sources = {phase: {'code.c': self.source} for phase in ('before', 'after')}
        self.requests = []
        args = argparse.Namespace(pair='fourier-horner', trials=3, seed=20261009,
                                  model='gpt-oss:20b', expected_digest=pilot.BASELINE_DIGEST,
                                  endpoint='http://127.0.0.1:11434/api/chat', output=self.output)

        def request(endpoint, data=None, timeout=30):
            if endpoint.endswith('/version'):
                return {'version': version}
            if endpoint.endswith('/show'):
                return {'details': {'family': 'gpt-oss'}}
            self.requests.append(copy.deepcopy(data))
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 420)
            if fail:
                raise TimeoutError('fixture infrastructure failure')
            text = (first if first is not None else self.malformed) if len(data['messages']) == 2 else (
                repaired if repaired is not None else self.valid)
            return {'done': True, 'done_reason': 'stop', 'message': {'content': text}}

        with patch.object(base, 'read_sources', return_value={'code.c': self.source}), \
                patch.object(base, 'model_identity', return_value={'digest': pilot.BASELINE_DIGEST}), \
                patch.object(pilot, 'request', side_effect=request), patch('builtins.print'):
            result = pilot.run_bounded_repair(args)
        path = self.output / 'responses.jsonl'
        self.rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return result

    def test_one_feedback_turn_and_both_responses_retained(self):
        self.assertEqual(self.run_fixture(), 0)
        self.assertEqual(len(self.requests), 12)
        self.assertEqual(len(self.rows), 12)
        self.assertEqual({row['kind'] for row in self.rows}, {'edit'})
        for first, repair in zip(self.rows[::2], self.rows[1::2]):
            self.assertEqual(first['attempt'], 'first')
            self.assertEqual(repair['attempt'], 'repair')
            self.assertEqual(first['response']['message']['content'], self.malformed)
            self.assertEqual(repair['response']['message']['content'], self.valid)
            self.assertEqual(repair['parent_response_sha256'], first['response_sha256'])
            self.assertEqual(first['settings'], repair['settings'])
            self.assertEqual(repair['request']['messages'][:2], first['request']['messages'])
            feedback = repair['request']['messages'][-1]['content']
            self.assertIn(first['score']['error'], feedback)
            self.assertIn('OLD_START,OLD_COUNT', feedback)
            self.assertNotIn('output_cartesian', feedback)
            self.assertNotIn('static 2', feedback)
            self.assertNotIn(self.valid, feedback)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertIsNone(summary['semantic_correctness'])

    def test_second_bad_patch_has_no_third_attempt(self):
        self.assertEqual(self.run_fixture(repaired=self.malformed), 0)
        self.assertEqual(len(self.requests), 12)
        self.assertTrue(all(row['score']['applies'] is False for row in self.rows))
        self.assertTrue(all(row['score']['semantic_correctness'] is None for row in self.rows))

    def test_applicable_first_patch_does_not_get_feedback(self):
        self.assertEqual(self.run_fixture(first=self.valid), 0)
        self.assertEqual(len(self.requests), 6)
        self.assertTrue(all(row['attempt'] == 'first' for row in self.rows))

    def test_infrastructure_failure_is_retained_and_stops(self):
        self.assertEqual(self.run_fixture(fail=True), 2)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.rows[0]['status'], 'INFERENCE_ERROR')
        self.assertTrue((self.output / 'blocked.json').is_file())
        self.assertFalse((self.output / 'summary.json').exists())

    def test_runtime_change_blocks_before_any_model_call(self):
        self.assertEqual(self.run_fixture(version='different'), 2)
        self.assertEqual(len(self.requests), 0)
        self.assertTrue((self.output / 'blocked.json').is_file())

    def test_replay_rejects_changed_feedback_and_third_attempt(self):
        self.assertEqual(self.run_fixture(), 0)
        changed = copy.deepcopy(self.rows)
        changed[1]['request']['messages'][-1]['content'] += '\nExtra assistance.'
        changed[1]['request_sha256'] = pilot.content_hash(changed[1]['request'])
        with self.assertRaisesRegex(ValueError, 'request or settings changed'):
            pilot.audit_repair_rows(changed, self.pair, self.sources)
        with self.assertRaisesRegex(ValueError, 'duplicate repair'):
            pilot.audit_repair_rows(self.rows + [self.rows[1]], self.pair, self.sources)

    def test_replay_rejects_wrong_parent_and_model(self):
        self.assertEqual(self.run_fixture(), 0)
        changed = copy.deepcopy(self.rows)
        changed[1]['parent_response_sha256'] = 'another response'
        with self.assertRaisesRegex(ValueError, 'rejected first response'):
            pilot.audit_repair_rows(changed, self.pair, self.sources)
        changed = copy.deepcopy(self.rows)
        changed[1]['model']['digest'] = 'another model'
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            pilot.audit_repair_rows(changed, self.pair, self.sources)

    def test_missing_repair_is_incomplete_not_success(self):
        self.assertEqual(self.run_fixture(), 0)
        with self.assertRaisesRegex(ValueError, 'Incomplete bounded repair run'):
            pilot.audit_repair_rows(self.rows[:-1], self.pair, self.sources)

    def test_successful_patch_cannot_be_given_repair_feedback(self):
        with self.assertRaises(ValueError):
            pilot.repair_feedback({'applies': True})


if __name__ == '__main__':
    unittest.main()
