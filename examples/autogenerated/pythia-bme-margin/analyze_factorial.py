import csv
import json
import sys
from pathlib import Path

def analyze(root):
    rows=list(csv.DictReader((root/'scores.tsv').open(),delimiter='\t'))
    result=[]
    for dest in ('step512','step1000'):
        for early in ('native','donor','zero'):
            for late in ('0','1'):
                cell=[r for r in rows if r['destination']==dest and r['early']==early and r['late_donor']==late]
                record={'destination':dest,'early':early,'late_donor':late}
                for prompt in ('bme_original','bme_heldout','unrelated_school_question'):
                    pairs=[]
                    for number in ('1','2','3'):
                        selected={r['candidate']:float(r['score']) for r in cell if r['prompt']==prompt and r['pair']==number}
                        pairs.append(selected['school']-selected['check'])
                    record[prompt]={'pairs':pairs,'mean':sum(pairs)/3}
                neutral=[float(r['score_change']) for r in cell if r['prompt'].startswith('neutral_')]
                record['neutral_mean_logp_change']=sum(neutral)/3
                record['mean_logit_delta_l2']=sum(float(r['logit_delta_l2']) for r in cell)/len(cell)
                record['mean_block16_activation_delta_l2']=sum(float(r['block16_last_prompt_delta_l2']) for r in cell)/len(cell)
                result.append(record)
    return result

if __name__=='__main__':
    print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
