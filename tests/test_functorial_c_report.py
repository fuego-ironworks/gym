from __future__ import annotations
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import functorial_c_eval as base
import functorial_c_pilot as pilot
import functorial_c_report as report

class ReportTests(unittest.TestCase):
    # Mock protocol records test the auditor, never counted as model evidence.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'protocol-test'
        self.bundle.mkdir()
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{}')
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        task = {'id': 'q', 'question': 'Return the supplied integer.', 'answer': 2}
        self.pair = {'id': 'protocol-test', 'repository': 'example/protocol',
                     'questions': [task], 'edits': [],
                     'before': {'commit': 'a'*40, 'files': []},
                     'after': {'commit': 'b'*40, 'files': []}}
        header = {'protocol': pilot.PROTOCOL, 'manifest_sha256': self.digest,
                  'model': {'digest': 'model-digest'}, 'runtime': {'version': 'test'}}
        (self.bundle / 'run.json').write_text(json.dumps(header))
        self.rows = []
        for phase in ('before', 'after'):
            request = {'messages': [{'role': 'system', 'content': pilot.SYSTEM},
                                   {'role': 'user', 'content': base.make_prompt({},task,'question')}],
                       'options': {'temperature': 0}, 'think': 'low'}
            self.rows.append({'protocol': pilot.PROTOCOL, 'pair': self.pair['id'],
                              'repository': self.pair['repository'], 'kind': 'question',
                              'task_id': 'q', 'trial': 0, 'phase': phase,
                              'commit': self.pair[phase]['commit'], 'source_files': [],
                              'manifest_sha256': self.digest, 'model': header['model'],
                              'settings': {'temperature': 0, 'think': 'low'},
                              'request': request, 'status': 'completed',
                              'response': {'done': True, 'done_reason': 'stop',
                                           'message': {'content': '{"answer":2}'}},
                              'score': {'parseable': True, 'correct': True}})

    def tearDown(self):
        self.temp.cleanup()

    def audit(self, rows, trials=1):
        (self.bundle / 'responses.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
        with patch.object(base,'MANIFEST',self.manifest), \
             patch.object(base,'read_manifest',return_value={'pairs':[self.pair]}), \
             patch.object(base,'read_sources',return_value={}):
            return report.audit(self.bundle,trials,True)

    def test_complete_pair_and_recomputed_scores(self):
        result = self.audit(self.rows)
        self.assertTrue(result['complete'])
        self.assertEqual(result['paired']['both_correct'],1)
        self.assertEqual(result['by_phase']['after']['correct_questions'],1)

    def test_unattempted_trials_are_reported_missing(self):
        result = self.audit(self.rows,trials=3)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['missing_responses']),4)
        self.assertEqual(result['planned_responses'],6)

    def test_false_score_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['response']['message']['content'] = '{"answer":99}'
        with self.assertRaisesRegex(ValueError,'score'):
            self.audit(rows)

    def test_altered_prompt_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['request']['messages'][1]['content'] = 'Replacement source'
        with self.assertRaisesRegex(ValueError,'Prompt'):
            self.audit(rows)

    def test_duplicate_records_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.audit(self.rows+self.rows[:1])

if __name__ == '__main__':
    unittest.main()
