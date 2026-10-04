"""Convert a ready reproduction and retain exact BME/control scores.

No provider client; analysis precedes any local retirement. Released anchors
add full tensor statistics and fixed-probe losses. Every checkpoint retains
the original, held-out and unrelated pair scores plus neutral-language losses.
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'llm/models/reproduction/pythia-410m-deduped'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'inspection'))
import verify_anchor
import checkpoint_retire
import make_dense_config

ANCHORS={512:'c63285838d79c704d97d3ef94674c47bd33aa17f',
         1000:'a3b3aff9a656ab34fec3474eb60bb5b487639539'}
TOKENIZER='c66f7467608ffee8fca0d28cf1f46a7574b53cec'
# Conservative comparison flags, not proof that an unsaved original path is recovered.
TOLERANCES={'relative_l2':0.001,'probe_loss_abs':0.02,'pair_margin_abs':0.02}

def evaluate(model,tokenizer,bme):
    pairs=[]
    for name in ('bme_original','bme_heldout','unrelated_school_question'):
        for i,(school,check) in enumerate(bme.score_pairs(model,tokenizer,bme.PROMPTS[name]),1):
            pairs.append({'prompt':name,'pair':i,'school':school['mean'],'check':check['mean'],
                          'margin':school['mean']-check['mean'],
                          'boundary_agrees':school['boundary_agrees'] and check['boundary_agrees']})
    neutral=[bme.score_one(model,tokenizer,p,c)['mean'] for p,c in bme.NEUTRAL]
    return {'pairs':pairs,'neutral_mean_logp':neutral}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--config-file',type=Path,required=True)
    parser.add_argument('--neox-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import bme_causal_recheck as bme
    step=checkpoint_retire.checkpoint_step(args.checkpoint_dir)
    if step not in make_dense_config.checkpoint_schedule(8):
        raise ValueError('checkpoint is outside the preregistered stride-8 schedule')
    files=checkpoint_retire.file_manifest(args.checkpoint_dir)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(bme.MODEL,revision=TOKENIZER)
    with tempfile.TemporaryDirectory(prefix='dense-analysis-') as directory:
        converted=Path(directory)
        verify_anchor.convert_checkpoint(args.neox_dir,args.checkpoint_dir,args.config_file,converted)
        local=AutoModelForCausalLM.from_pretrained(converted,torch_dtype=torch.float32).eval()
        behavior=evaluate(local,tokenizer,bme)
        result={'step':step,'trajectory':'reproduction, not an unreleased EleutherAI checkpoint',
                'source_files':[r.__dict__ for r in files],'config_sha256':verify_anchor.sha256_file(args.config_file),
                'tokenizer_sha':TOKENIZER,'behavior':behavior,'tolerances':TOLERANCES,'anchor':None,
                'torch':torch.__version__}
        if step in ANCHORS:
            official=AutoModelForCausalLM.from_pretrained(bme.MODEL,revision=ANCHORS[step],torch_dtype=torch.float32).eval()
            stats=verify_anchor.tensor_error_stats(local.state_dict(),official.state_dict())
            reference=evaluate(official,tokenizer,bme)
            neutral_loss_deltas=[a-b for a,b in zip(behavior['neutral_mean_logp'],reference['neutral_mean_logp'])]
            pair_deltas=[a['margin']-b['margin'] for a,b in zip(behavior['pairs'],reference['pairs'])]
            result['anchor']={'sha':ANCHORS[step],'tensor_error':stats,'behavior':reference,
                'neutral_logp_deltas':neutral_loss_deltas,'pair_margin_deltas':pair_deltas,
                'within_comparison_tolerances':stats['relative_l2']<=TOLERANCES['relative_l2']
                    and max(map(abs,neutral_loss_deltas))<=TOLERANCES['probe_loss_abs']
                    and max(map(abs,pair_deltas))<=TOLERANCES['pair_margin_abs']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(args.output)

if __name__=='__main__':
    main()
