"""Small causal factorial: layer-0 final token × downstream blocks 16–18.

Uses released anchors, canonical tokenization and pairs. Whole-block weight
transplants are potentially off-distribution; neutral damage remains explicit.
"""
import contextlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'llm/models/inspection'))
import bme_causal_recheck as original

REVISIONS={'step512':'c63285838d79c704d97d3ef94674c47bd33aa17f',
           'step1000':'a3b3aff9a656ab34fec3474eb60bb5b487639539'}
TOKENIZER='c66f7467608ffee8fca0d28cf1f46a7574b53cec'
WINDOW=(16,17,18)

@contextlib.contextmanager
def downstream(recipient,donor,replace):
    saved=[recipient.gpt_neox.layers[i] for i in WINDOW]
    try:
        if replace:
            for i in WINDOW:
                recipient.gpt_neox.layers[i]=donor.gpt_neox.layers[i]
        yield
    finally:
        for i,block in zip(WINDOW,saved):
            recipient.gpt_neox.layers[i]=block

def native(model,ids,start):
    captured=[]
    def capture(_module,_inputs,output):
        captured.append(output[:,start-1:start].detach().clone())
    handle=model.gpt_neox.layers[0].mlp.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            output=model(ids,use_cache=False,output_hidden_states=True)
        return output,captured[0]
    finally:
        handle.remove()

def edit(model,donor,ids,start,early,late,donor_value):
    def hook(_module,_inputs,output):
        result=output.clone()
        result[:,start-1:start]=0 if early=='zero' else donor_value
        return result
    handle=model.gpt_neox.layers[0].mlp.register_forward_hook(hook) if early!='native' else None
    try:
        with downstream(model,donor,late),torch.inference_mode():
            return model(ids,use_cache=False,output_hidden_states=True)
    finally:
        if handle:
            handle.remove()

def logp(output,ids,start):
    return output.logits[0,start-1:-1].float().log_softmax(-1).gather(-1,ids[0,start:,None]).squeeze(-1)

def main():
    output_dir=Path(sys.argv[1])
    sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no']):
        raise RuntimeError('tracked source must be clean')
    output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(original.MODEL,revision=TOKENIZER)
    models={name:AutoModelForCausalLM.from_pretrained(original.MODEL,revision=ref,
        torch_dtype=torch.float32,attn_implementation='eager').eval() for name,ref in REVISIONS.items()}
    cases=[]
    for prompt in ('bme_original','bme_heldout','unrelated_school_question'):
        for pair,choices in enumerate(original.PAIRS,1):
            for candidate,text in zip(('school','check'),choices):
                cases.append((prompt,pair,candidate,original.PROMPTS[prompt],text))
    for i,(prompt,text) in enumerate(original.NEUTRAL,1):
        cases.append((f'neutral_{i}','','',prompt,text))
    rows=[]
    for destination,model in models.items():
        donor_name=next(n for n in models if n!=destination)
        donor=models[donor_name]
        for prompt,pair,candidate,prefix,text in cases:
            ids,start,boundary=original.encode(tokenizer,prefix,text)
            baseline,own_value=native(model,ids,start)
            _,donor_value=native(donor,ids,start)
            base_logp=logp(baseline,ids,start)
            self_output=edit(model,model,ids,start,'donor',True,own_value)
            torch.testing.assert_close(self_output.logits,baseline.logits,rtol=0,atol=1e-5)
            for early in ('native','donor','zero'):
                for late in (False,True):
                    result=baseline if early=='native' and not late else edit(model,donor,ids,start,early,late,donor_value)
                    scores=logp(result,ids,start)
                    row={'destination':destination,'donor':donor_name,'prompt':prompt,'pair':pair,
                        'candidate':candidate,'early':early,'late_donor':int(late),
                        'baseline_score':base_logp.mean().item(),'score':scores.mean().item(),
                        'score_change':(scores-base_logp).mean().item(),'boundary_agrees':boundary,
                        'logit_delta_l2':(result.logits[0,start-1:-1]-baseline.logits[0,start-1:-1]).double().norm().item(),
                        'block16_last_prompt_delta_l2':(result.hidden_states[17][:,start-1]-baseline.hidden_states[17][:,start-1]).double().norm().item(),
                        'final_last_prompt_delta_l2':(result.hidden_states[-1][:,start-1]-baseline.hidden_states[-1][:,start-1]).double().norm().item()}
                    rows.append(row)
            original.tsv(output_dir/'scores.tsv',tuple(rows[0]),rows)
            print(destination,prompt,pair,candidate,flush=True)
    manifest={'source_sha':sha,'model':original.MODEL,'checkpoints':REVISIONS,
        'tokenizer_sha':TOKENIZER,'window':WINDOW,'early_token':'final prompt token only',
        'late_intervention':'recipient blocks 16–18 replaced with donor blocks; other parameters native',
        'prompts':original.PROMPTS,'pairs':original.PAIRS,'neutral':original.NEUTRAL,
        'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),
        'provider':'local CPU','paid_execution_allowed':False,'self_patch_passed':True}
    (output_dir/'COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    main()
