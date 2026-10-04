"""Independently re-evaluate final saved steps with a narrow central difference.

Large 0.05 path fractions are not local on unstable updates; preserve those
receipts, and check the saved JVP with a 0.0001 fraction in float64 instead.
"""
import csv
import json
import sys
from pathlib import Path
import run as experiment
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

rows=[]
torch.set_num_threads(2)
for argument in sys.argv[1:]:
    root=Path(argument)
    manifest=json.loads((root/'COMPLETE.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['model_revision'])
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['model_revision'],
        torch_dtype=torch.float64,attn_implementation='eager').eval()
    probes=experiment.probes(tokenizer)[:7]
    for case,layer in (('layer0_one',0),('layer0_two',0),('layer2_two',2)):
        target=f'gpt_neox.layers.{layer}.attention.query_key_value'
        adapter=experiment.attach_lora(model,target,seed=manifest['seed'])
        payload_path=root/case/'step-0008.pt'
        payload=torch.load(payload_path,map_location='cpu',weights_only=True)
        a,b,da,db=(payload[k].double() for k in ('a_before','b_before','delta_a','delta_b'))
        torch.set_default_dtype(torch.float64)
        try:
            for name,pair,candidate,ids,start in probes:
                row,_=experiment.compare_probe(model,adapter,target,ids,start,a,b,da,db)
                rows.append({'run':root.name,'case':case,'probe':name,'saved_step_sha256':experiment.file_sha256(payload_path),
                    'central_fraction':0.0001,**row})
        finally:
            torch.set_default_dtype(torch.float32)
            parent=model.get_submodule(f'gpt_neox.layers.{layer}.attention')
            parent.query_key_value=adapter.base
    del model
experiment.write_table(Path(sys.argv[1]).parent/'narrow-tangent-check.tsv',rows)
print(json.dumps([{'run':r['run'],'case':r['case'],'probe':r['probe'],
       'fd_relative_error':r['jvp_fd_error_l2']/r['logit_tangent_l2'],
       'nonlinear_residual_fraction':r['logit_residual_l2']/r['logit_actual_l2']} for r in rows],indent=2))
