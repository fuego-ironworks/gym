"""Exercise real provenance validator without loading neural dependencies."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'llm/models/inspection/bme_tuned_lens.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_checkpoint_lens')
scope={'Path':Path,'json':json,'MODEL':'EleutherAI/pythia-410m-deduped'}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)

class LensProvenanceTests(unittest.TestCase):
    def check(self,revision,unembed_hash):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'params.pt').write_bytes(b'provenance fixture only')
            (path/'config.json').write_text(json.dumps({'base_model_name_or_path':scope['MODEL'],
                'base_model_revision':revision,'unembed_hash':unembed_hash}))
            return scope['read_checkpoint_lens'](path,'step512','c'*40)
    def test_missing_hash_and_mutable_revision_fail(self):
        for revision,hash_ in [('step512','a'*64),('c'*40,None),('c'*40,''),('d'*40,'a'*64)]:
            with self.subTest(revision=revision,hash=hash_), self.assertRaises(RuntimeError):
                self.check(revision,hash_)
    def test_exact_identity_and_hash_are_required(self):
        config,_=self.check('c'*40,'a'*64)
        self.assertEqual('c'*40,config['base_model_revision'])

if __name__=='__main__':
    unittest.main()
