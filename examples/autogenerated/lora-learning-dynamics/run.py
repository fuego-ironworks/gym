"""Tests 3–4: compare adjacent LoRA tangents with actual held-out behavior.

CPU only. Full A/B and weight decomposition retained; probe tensors are
summarized plus exact target-token differential vectors. No provider client.
"""
import argparse
import csv
import copy
import json
import os
import platform
import re
import sys
from pathlib import Path
import torch
import transformers
from huggingface_hub import HfApi
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'llm/models'))
sys.path.insert(0, str(ROOT / 'llm/models/inspection'))
from lora_differential import attach_lora, frozen_parameter_hashes, named_hashes_digest
from run_pythia_14m_lora_inspectability import source_receipt, file_sha256
import bme_causal_recheck as bme

TRAIN = ('The moon orbits the Earth.', 'Mars orbits the Sun.')
TEXTS = {
    'train_moon': TRAIN[0], 'train_mars': TRAIN[1],
    'heldout_moon': 'The Moon travels around the Earth.',
    'heldout_mars': 'The planet Mars travels around the Sun.',
    'neutral_library': 'The library opened at nine on Tuesday. The first visitor returned a book.',
    'neutral_carpenter': 'A carpenter measured the wooden board twice. Then she cut it to length.',
    'neutral_rain': 'The rain ended before the afternoon walk. The streets dried in the sun.',
}

def decompose(a, b, da, db, scale=1.0):
    return tuple(scale * x for x in (b @ da, db @ a, db @ da))

def write_table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def probes(tokenizer):
    result = []
    for name, text in TEXTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        result.append((name, '', '', torch.tensor([ids]), 1))
    for name in ('bme_original', 'bme_heldout', 'unrelated_school_question'):
        for pair, choices in enumerate(bme.PAIRS, 1):
            for candidate, continuation in zip(('school', 'check'), choices):
                ids, start, agrees = bme.encode(tokenizer, bme.PROMPTS[name], continuation)
                result.append((name, pair, candidate, ids, start))
    return result

def score(logits, ids, start):
    selected = logits[0, start-1:-1].log_softmax(-1)
    return selected.gather(-1, ids[0, start:, None]).squeeze(-1)

def score_tangent(logits, tangent, ids, start):
    selected = logits[0, start-1:-1]
    direction = tangent[0, start-1:-1]
    logp_direction = direction - (selected.softmax(-1) * direction).sum(-1, keepdim=True)
    return logp_direction.gather(-1, ids[0, start:, None]).squeeze(-1)

def compare_probe(model, adapter, target, ids, start, a, b, da, db):
    def output_at(fraction):
        params = {target+'.a': a + fraction * da, target+'.b': b + fraction * db}
        output = functional_call(model, params, (), {'input_ids': ids, 'use_cache': False,
                                  'output_hidden_states': True})
        return (output.logits, *output.hidden_states[1:])
    point = torch.zeros((), dtype=a.dtype)
    before, tangent = torch.autograd.functional.jvp(output_at, point, torch.ones_like(point))
    with torch.no_grad():
        after = output_at(torch.ones_like(point))
        epsilon = 0.0001 if a.dtype==torch.float64 else 0.05
        minus, plus = output_at(point-epsilon), output_at(point+epsilon)
    actual = tuple(y-x for x,y in zip(before,after))
    residual = tuple(x-y for x,y in zip(actual,tangent))
    finite = (plus[0]-minus[0])/(2*epsilon)
    before_score, after_score = score(before[0], ids, start), score(after[0], ids, start)
    predicted = score_tangent(before[0], tangent[0], ids, start)
    delta = after_score-before_score
    norm=lambda x:x.detach().double().norm().item()
    combined=lambda xs:sum(norm(x)**2 for x in xs)**0.5
    row = {'score_before': before_score.mean().item(), 'score_after': after_score.mean().item(),
           'score_change': delta.mean().item(), 'score_tangent': predicted.mean().item(),
           'score_residual': (delta-predicted).mean().item(),
           'logit_actual_l2': norm(actual[0]), 'logit_tangent_l2': norm(tangent[0]),
           'logit_residual_l2': norm(residual[0]), 'jvp_fd_error_l2': norm(finite-tangent[0]),
           'activation_actual_l2': combined(actual[1:]),
           'activation_tangent_l2': combined(tangent[1:]),
           'activation_residual_l2': combined(residual[1:])}
    vectors = {'score_before':before_score, 'score_after':after_score, 'score_tangent':predicted,
               'layer_actual_l2':torch.tensor([norm(x) for x in actual[1:]]),
               'layer_tangent_l2':torch.tensor([norm(x) for x in tangent[1:]]),
               'layer_residual_l2':torch.tensor([norm(x) for x in residual[1:]])}
    return row, {k:v.detach().cpu() for k,v in vectors.items()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--model-revision', default='0412c27fd5dcade8d1ff02273769acfb0c71421b')
    args=parser.parse_args()
    if not 3 <= args.steps <= 16:
        raise ValueError('bounded experiment needs 3–16 steps')
    source=source_receipt(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model_name='EleutherAI/pythia-14m-deduped'
    revision=args.model_revision
    if re.fullmatch('[0-9a-f]{40}',revision) is None:
        raise ValueError('immutable model revision required')
    tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision)
    all_probes=probes(tokenizer)
    rows, weights=[],[]
    # Test 3 varies site at fixed data. Test 4 varies data at fixed site.
    cases=(('layer0_two',0,TRAIN), ('layer2_two',2,TRAIN), ('layer0_one',0,TRAIN[:1]))
    for case, layer, train_texts in cases:
        torch.manual_seed(args.seed)
        model=AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                torch_dtype=torch.float32, attn_implementation='eager').eval()
        target=f'gpt_neox.layers.{layer}.attention.query_key_value'
        adapter=attach_lora(model,target,seed=args.seed)
        # Keep training float32. Evaluate the same saved weights in float64 so
        # tiny finite differences are not swamped by float32 subtraction noise.
        probe_model=copy.deepcopy(model).double()
        probe_adapter=probe_model.get_submodule(target)
        frozen=frozen_parameter_hashes(model)
        optimizer=torch.optim.SGD([adapter.a,adapter.b],lr=args.learning_rate)
        case_dir=args.output/case
        case_dir.mkdir()
        for step in range(1,args.steps+1):
            a,b=adapter.a.detach().clone(),adapter.b.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for text in train_texts:
                ids=tokenizer(text,return_tensors='pt',add_special_tokens=False)['input_ids']
                losses.append(model(input_ids=ids,labels=ids,use_cache=False).loss)
            train_loss=torch.stack(losses).mean()
            train_loss.backward()
            grad_a,grad_b=adapter.a.grad.detach().clone(),adapter.b.grad.detach().clone()
            optimizer.step()
            da,db=adapter.a.detach()-a,adapter.b.detach()-b
            terms=decompose(a,b,da,db,adapter.scaling)
            delta=adapter.delta_weight().detach()-adapter.scaling*(b@a)
            reconstruction=delta-sum(terms)
            torch.testing.assert_close(delta,sum(terms),rtol=2e-4,atol=2e-8)
            if step==1:
                assert torch.count_nonzero(grad_a)==0 and torch.count_nonzero(da)==0
            assert frozen==frozen_parameter_hashes(model)
            payload={'a_before':a,'b_before':b,'a_after':adapter.a.detach().clone(),
                'b_after':adapter.b.detach().clone(),'grad_a':grad_a,'grad_b':grad_b,
                'delta_a':da,'delta_b':db,'B_deltaA':terms[0],'deltaB_A':terms[1],
                'deltaB_deltaA':terms[2],'delta_weight':delta,'probes':{}}
            weight={'case':case,'step':step,'train_loss_before':train_loss.item(),
                'grad_a_l2':grad_a.double().norm().item(),'grad_b_l2':grad_b.double().norm().item(),
                'B_deltaA_l2':terms[0].double().norm().item(),
                'deltaB_A_l2':terms[1].double().norm().item(),
                'deltaB_deltaA_l2':terms[2].double().norm().item(),
                'reconstruction_error_l2':reconstruction.double().norm().item()}
            weights.append(weight)
            for name,pair,candidate,ids,start in all_probes:
                torch.set_default_dtype(torch.float64)
                try:
                    result,vectors=compare_probe(probe_model,probe_adapter,target,ids,start,
                        a.double(),b.double(),da.double(),db.double())
                finally:
                    torch.set_default_dtype(torch.float32)
                rows.append({'case':case,'step':step,'probe':name,'pair':pair,
                             'candidate':candidate,**result})
                payload['probes'][f'{name}:{pair}:{candidate}']=vectors
            torch.save(payload,case_dir/f'step-{step:04}.pt')
            write_table(args.output/'weights.tsv',weights)
            write_table(args.output/'probes.tsv',rows)
            print(case,step,'loss',train_loss.item(),'mixed_l2',weight['deltaB_deltaA_l2'],flush=True)
        del model,adapter,optimizer,probe_model,probe_adapter
    manifest={'source':source,'model':model_name,'model_revision':revision,
        'steps':args.steps,'seed':args.seed,'optimizer':'SGD','learning_rate':args.learning_rate,
        'adapter':{'rank':1,'alpha':1,'initialization':'Kaiming A, zero B'},
        'train_texts':TRAIN,'probe_texts':TEXTS,
        'bme_prompts':{n:bme.PROMPTS[n] for n in ('bme_original','bme_heldout','unrelated_school_question')},
        'bme_pairs':bme.PAIRS,'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
        'python':platform.python_version(),'threads':2,
        'provider':'GitHub CPU' if os.environ.get('GITHUB_ACTIONS')=='true' else 'local CPU',
        'github_run_id':os.environ.get('GITHUB_RUN_ID'),
        'github_run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT'),
        'training_precision':'float32','probe_precision':'float64 from saved float32 weights'},
        'paid_execution_allowed':False,'frozen_base':named_hashes_digest(frozen),
        'artifacts':{str(p.relative_to(args.output)):file_sha256(p) for p in args.output.rglob('*') if p.is_file()}}
    (args.output/'COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    main()
