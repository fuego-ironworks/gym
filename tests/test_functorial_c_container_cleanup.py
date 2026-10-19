from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from functorial_c_semantics import evaluate

class ContainerCleanupTests(unittest.TestCase):
    def evaluate_with(self, first, execution=None):
        with tempfile.TemporaryDirectory() as directory:
            compiler=Path(directory)/'bin/ick'
            compiler.parent.mkdir(); compiler.write_text('not executed: mock fixture')
            outcomes=[first]
            if execution is not None:
                outcomes.append(execution)
            outcomes.append(subprocess.CompletedProcess([],0))
            with patch('functorial_c_semantics.subprocess.run',side_effect=outcomes) as run:
                result=evaluate('int x;', 'int x;', compiler, 'sha256:fixture')
                calls=run.call_args_list
        self.assertEqual(len(calls),3 if execution is not None else 2)
        launch=calls[0].args[0]
        name=launch[launch.index('--name')+1]
        self.assertEqual(calls[-1].args[0],['docker','rm','--force',name])
        self.assertIn('--network=none',launch)
        self.assertIn('core=0',launch)
        if execution is not None:
            execute=calls[1].args[0]
            self.assertEqual(execute[-1],'./check')
            self.assertIn('--network=none',execute)
            self.assertLessEqual(calls[1].kwargs['timeout'],60)
        return result

    def test_timeout_removes_container(self):
        result=self.evaluate_with(subprocess.TimeoutExpired('docker',60))
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIsNone(result['semantic_correctness'])

    def test_unavailable_container_is_not_wrong_answer(self):
        result=self.evaluate_with(subprocess.CompletedProcess([],125,'','daemon error'))
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIsNone(result['semantic_correctness'])

    def test_success_still_cleans_up(self):
        result=self.evaluate_with(subprocess.CompletedProcess([],0,'compiled',''),
                                  subprocess.CompletedProcess([],0,'PASS',''))
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['stage'],'execute')

    def test_compile_failure_is_not_executed_semantics(self):
        result=self.evaluate_with(subprocess.CompletedProcess([],1,'','compiler error'))
        self.assertEqual(result['status'],'FAIL_COMPILE')
        self.assertEqual(result['stage'],'compile')
        self.assertIsNone(result['semantic_correctness'])
        self.assertEqual(len(result['stages']),1)

    def test_executed_assertion_is_distinct_from_compile_failure(self):
        result=self.evaluate_with(subprocess.CompletedProcess([],0,'compiled',''),
                                  subprocess.CompletedProcess([],134,'','assertion failed'))
        self.assertEqual(result['status'],'FAIL_EXECUTION')
        self.assertEqual(result['stage'],'execute')
        self.assertFalse(result['semantic_correctness'])
        self.assertEqual(result['stages'][0]['exit_code'],0)
        self.assertEqual(result['stages'][1]['exit_code'],134)

    def test_execution_timeout_cleans_up(self):
        result=self.evaluate_with(subprocess.CompletedProcess([],0,'compiled',''),
                                  subprocess.TimeoutExpired('docker',60))
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(result['stage'],'execute')
        self.assertIsNone(result['semantic_correctness'])

if __name__=='__main__':
    unittest.main()
