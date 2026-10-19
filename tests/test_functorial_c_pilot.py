from __future__ import annotations
import copy
import difflib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from functorial_c_pilot import strict_json, score_answer, outcome, paired_summary
from functorial_c_semantics import apply_patch, positive_control

class PilotTests(unittest.TestCase):
    def row(self, phase, correct=True):
        return {'pair':'test', 'kind':'question', 'task_id':'q', 'trial':0,
                'phase':phase, 'model':{'digest':'same'}, 'settings':{'seed':1},
                'status':'completed', 'score':{'correct':correct}}

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"answer":1,"answer":2}')

    def test_non_json_numbers_rejected(self):
        for value in ('NaN','Infinity','-Infinity'):
            with self.assertRaises(ValueError):
                strict_json('{"answer":'+value+'}')

    def test_boolean_is_not_number(self):
        self.assertFalse(score_answer('{"answer":true}', 1, 'count')['correct'])

    def test_float_and_literal_answers_both_accepted(self):
        expected={'x':1.9,'y':1.9,'z':0}
        self.assertTrue(score_answer('{"answer":{"x":1.899999976158142,"y":1.9,"z":0}}',
                                     expected,'corner-center')['correct'])
        self.assertFalse(score_answer('{"answer":{"x":1.89,"y":1.9,"z":0}}',
                                      expected,'corner-center')['correct'])

    def test_truncation_is_not_wrong_answer(self):
        self.assertEqual(outcome({'done':True,'done_reason':'length',
                                  'message':{'content':'{"answer":2}'}}),'OUTPUT_TRUNCATED')

    def test_missing_done_flag_rejected(self):
        self.assertEqual(outcome({'message':{'content':'{"answer":2}'}}),'INCOMPLETE_RESPONSE')

    def test_paired_wins_and_losses(self):
        report=paired_summary([self.row('before',False),self.row('after',True)])
        self.assertEqual(report['after_only'],1)
        self.assertEqual(report['before_only'],0)

    def test_missing_partner_is_incomplete(self):
        self.assertEqual(paired_summary([self.row('before')])['incomplete_pairs'],1)

    def test_duplicate_trial_not_double_counted(self):
        with self.assertRaises(ValueError):
            paired_summary([self.row('before'),self.row('before')])

    def test_model_change_rejected(self):
        after=self.row('after'); after['model']['digest']='different'
        with self.assertRaises(ValueError):
            paired_summary([self.row('before'),after])

    def test_sampling_change_rejected(self):
        after=self.row('after'); after['settings']['seed']=2
        with self.assertRaises(ValueError):
            paired_summary([self.row('before'),after])

class PatchTests(unittest.TestCase):
    # These tiny strings test patch machinery, not model comprehension.
    def diff(self, before, after):
        return ''.join(difflib.unified_diff(before.splitlines(keepends=True),
                    after.splitlines(keepends=True),fromfile='a/code.c',tofile='b/code.c'))

    def test_actual_hunk_counts_and_unicode(self):
        before='a ← 1;\nb ← 2;\n'; after='a ← 1;\nb ← 3;\n'
        self.assertEqual(apply_patch(before,self.diff(before,after)),after)

    def test_multiple_hunks(self):
        before=''.join('line %d\n'%i for i in range(30))
        after=before.replace('line 2\n','changed 2\n').replace('line 25\n','changed 25\n')
        self.assertEqual(apply_patch(before,self.diff(before,after)),after)

    def test_wrong_file_rejected(self):
        with self.assertRaises(ValueError):
            apply_patch('a\n',self.diff('a\n','b\n').replace('b/code.c','b/other.c'))

    def test_extra_git_directive_rejected(self):
        with self.assertRaises(ValueError):
            apply_patch('a\n',self.diff('a\n','b\n')+'diff --git a/x b/x\nnew file mode 120000\n')

    def test_stale_context_rejected(self):
        with self.assertRaises(ValueError):
            apply_patch('z\n',self.diff('a\n','b\n'))

    def test_wrong_line_counts_rejected(self):
        with self.assertRaises(ValueError):
            apply_patch('a\n',self.diff('a\n','b\n').replace('@@ -1 +1 @@','@@ -1,2 +1 @@'))

if __name__=='__main__':
    unittest.main()
