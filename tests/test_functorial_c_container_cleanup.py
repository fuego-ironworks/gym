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
    def evaluate_with(self, first):
        with tempfile.TemporaryDirectory() as directory:
            compiler=Path(directory)/'bin/ick'
            compiler.parent.mkdir(); compiler.write_text('not executed: mock fixture')
            with patch('functorial_c_semantics.subprocess.run',side_effect=[
                    first,subprocess.CompletedProcess([],0)]) as run:
                result=evaluate('int x;', 'int x;', compiler, 'sha256:fixture')
                calls=run.call_args_list
        self.assertEqual(len(calls),2)
        launch=calls[0].args[0]
        name=launch[launch.index('--name')+1]
        self.assertEqual(calls[1].args[0],['docker','rm','--force',name])
        self.assertIn('--network=none',launch)
        self.assertIn('core=0',launch)
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
        result=self.evaluate_with(subprocess.CompletedProcess([],0,'PASS',''))
        self.assertEqual(result['status'],'PASS')

if __name__=='__main__':
    unittest.main()
