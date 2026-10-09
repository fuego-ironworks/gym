#!/usr/bin/env python3
"""Execute the Horner edit under ICK in a networkless resource-limited container."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
import functorial_c_eval as base

DRIVER = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
extern void historical_polynomial(const double *, unsigned, unsigned, double, double, double *);
extern void fourier_polynomial_cartesian_ick(const double *, unsigned, unsigned, double, double, double *);
static int same(double a, double b) {
    return (isnan(a) && isnan(b)) || memcmp(&a, &b, sizeof(a)) == 0;
}
int main(void) {
    const double points[7][2] = {{0,0},{-0.0,-0.0},{1,0},{0.25,-0.5},{-0.75,0.4},{INFINITY,0},{NAN,1}};
    unsigned checks = 0;
    for (unsigned n=0; n<=6; ++n) for (unsigned terms=0; terms<=8; ++terms)
    for (unsigned p=0; p<7; ++p) for (unsigned alias=0; alias<2; ++alias) {
        double a[14] = {1,-0.0,2,3,-0.5,1,0.25,-0.75,-2,0.125,4,-1,12345,67890};
        double b[14]; memcpy(b,a,sizeof(a));
        double oa[4] = {12345,67890,11111,22222}, ob[4]; memcpy(ob,oa,sizeof(oa));
        double *outa=alias?a:oa, *outb=alias?b:ob;
        fourier_polynomial_cartesian_ick(a,n,terms,points[p][0],points[p][1],outa);
        historical_polynomial(b,n,terms,points[p][0],points[p][1],outb);
        for (unsigned i=0; i<14; ++i) assert(same(a[i],b[i]));
        for (unsigned i=0; i<4; ++i) assert(same(oa[i],ob[i]));
        ++checks;
    }
    const double c[4] = {1,2,3,-1}; double out[2];
    fourier_polynomial_cartesian_ick(c,2,2,0,1,out);
    assert(out[0]==2 && out[1]==5);
    fourier_polynomial_cartesian_ick(0,0,7,NAN,NAN,out);
    assert(out[0]==0 && out[1]==0);
    /* Null output must return before even reading the null input. */
    fourier_polynomial_cartesian_ick(0,2,2,NAN,NAN,0);
    fourier_polynomial_cartesian_ick(0,0,7,NAN,NAN,0);
    assert(checks == 882);
    puts("PASS: 882 historical comparisons, independent value, sentinels, null input/output");
    return 0;
}
'''


def apply_patch(original, patch):
    """Only a text-only unified code.c patch; no Git metadata can create other files."""
    if len(patch.encode()) > 100_000:
        raise ValueError('Patch too large')
    patch = patch.strip('\r\n')
    fence = chr(96)*3
    if patch.startswith(fence) and patch.endswith(fence):
        lines = patch.splitlines()
        if lines[0] not in (fence, fence+'diff', fence+'patch'):
            raise ValueError('Unsupported fence')
        patch = '\n'.join(lines[1:-1])
    lines = patch.splitlines(keepends=True)
    lines = [line if line.endswith('\n') else line+'\n' for line in lines]
    if lines and lines[0].rstrip() == 'diff --git a/code.c b/code.c':
        lines.pop(0)
        if lines and re.fullmatch(r'index [0-9a-f]+\.\.[0-9a-f]+(?: 100644)?\n', lines[0]):
            lines.pop(0)
    if lines[:2] != ['--- a/code.c\n','+++ b/code.c\n']:
        raise ValueError('Only code.c text changes are permitted')
    lines = lines[2:]
    source = original.splitlines(keepends=True)
    output, position, hunks = [], 0, 0
    while lines:
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n', lines.pop(0))
        if not match:
            raise ValueError('Expected a unified hunk')
        old_start, old_count, new_start, new_count = match.groups()
        old_count, new_count = int(old_count or 1), int(new_count or 1)
        start = int(old_start) - (1 if old_count else 0)
        if start < position or start > len(source):
            raise ValueError('Invalid or overlapping old range')
        output.extend(source[position:start]); position = start
        wanted_new = int(new_start) - (1 if new_count else 0)
        if wanted_new != len(output):
            raise ValueError('Incorrect new hunk position')
        old_seen = new_seen = 0
        while lines and not lines[0].startswith('@@ '):
            line = lines.pop(0)
            prefix, value = line[:1], line[1:]
            if prefix not in (' ', '-', '+'):
                raise ValueError('Non-text or unexpected patch directive')
            if prefix != '+':
                if position >= len(source) or source[position] != value:
                    raise ValueError('Context differs from pinned source')
                position += 1; old_seen += 1
            if prefix != '-':
                output.append(value); new_seen += 1
        if (old_seen,new_seen) != (old_count,new_count):
            raise ValueError('Incorrect unified hunk counts')
        hunks += 1
    if not hunks:
        raise ValueError('Empty patch')
    output.extend(source[position:])
    result = ''.join(output)
    if result == original:
        raise ValueError('No-op patch')
    return result


def positive_control(source):
    source = source.replace('output_cartesian[static 2]', 'output_cartesian[2]')
    found = re.search(r'void\s+fourier_polynomial_cartesian_ick\s*\([^)]*\)\s*\{', source)
    if not found:
        raise ValueError('Historical function boundary missing')
    return source[:found.end()] + '\n    if (!output_cartesian) return;\n' + source[found.end():]


def evaluate(candidate, reference, compiler, image):
    if re.search(r'output_cartesian\s*\[\s*static', candidate):
        return {'status': 'FAIL_CONTRACT', 'stage': 'source_contract',
                'semantic_correctness': None, 'contract_satisfied': False}
    stage = compiler.resolve().parent.parent
    container_name = 'gym-ick-edit-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='gym-ick-edit-') as name:
        work = Path(name)
        (work/'candidate.c').write_text(candidate)
        (work/'reference.c').write_text(reference)
        (work/'driver.c').write_text(DRIVER)
        script = ('set -eu; "$1" -fno-link-libatomic -std=c17 -O2 -Wall -Wextra -Werror '
                  '-Dfourier_polynomial_cartesian_ick=historical_polynomial -c reference.c -o reference.o; '
                  '"$1" -fno-link-libatomic -std=c17 -O2 -Wall -Wextra -Werror '
                  'candidate.c driver.c reference.o -lm -o check')
        command = ['docker','run','--rm','--name',container_name,'--network=none',
                   '--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
                   '--memory=1g','--cpus=2','--pids-limit=64','--ulimit','core=0',
                   '--user',f'{os.getuid()}:{os.getgid()}',
                   '--tmpfs','/tmp:rw,exec,nosuid,nodev,size=128m',
                   '--mount',f'type=bind,src={stage},dst={stage},readonly',
                   '--mount',f'type=bind,src={work},dst=/work','--workdir=/work',
                   image]
        stages = []
        started = time.monotonic()
        stage = 'compile'
        try:
            for stage, arguments in (
                    ('compile',['/bin/sh','-c',script,'check',str(compiler.resolve())]),
                    ('execute',['./check'])):
                remaining = 60 - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command,60)
                run = subprocess.run(command+arguments, capture_output=True, text=True, timeout=remaining)
                stages.append({'stage':stage, 'exit_code':run.returncode,
                               'stdout':run.stdout[-6000:], 'stderr':run.stderr[-6000:]})
                if run.returncode == 0 and stage == 'compile':
                    continue
                unavailable = run.returncode in (125,126,127)
                return {'status':'BLOCKED' if unavailable else
                                'FAIL_COMPILE' if stage == 'compile' else
                                'PASS' if run.returncode == 0 else 'FAIL_EXECUTION',
                        'semantic_correctness':None if unavailable or stage == 'compile'
                                               else run.returncode == 0,
                        'stage':stage, 'stages':stages,
                        'exit_code':run.returncode,'stdout':run.stdout[-6000:],
                        'stderr':run.stderr[-6000:]}
        except (OSError,subprocess.TimeoutExpired) as exc:
            return {'status':'BLOCKED','semantic_correctness':None,'error':repr(exc),
                    'stage':stage,'stages':stages}
        finally:
            # Killing only the Docker client need not stop its server-side container.
            # Remove the named container before deleting the writable work directory.
            try:
                subprocess.run(['docker','rm','--force',container_name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10, check=False)
            except (OSError,subprocess.TimeoutExpired):
                pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--image',required=True,help='Resolved local Docker image SHA, not a moving tag')
    parser.add_argument('--receipts',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.image.startswith('sha256:') or not args.compiler.is_file():
        parser.error('Pinned ICK compiler and resolved image digest required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pair=next(p for p in base.read_manifest()['pairs'] if p['id']=='fourier-horner')
    sources={phase:base.read_sources(pair,phase,base.ROOT/'.cache/functorial-c',False)['code.c']
             for phase in ('before','after')}
    results=[]
    for phase,source in sources.items():
        good=positive_control(source)
        changed=good.replace('output_cartesian[0]', 'output_cartesian[1]')
        positive=evaluate(good,source,args.compiler,args.image)
        negative=evaluate(changed,source,args.compiler,args.image)
        results.append({'kind':'control','phase':phase,'positive':positive,'negative':negative})
        if (positive['status']!='PASS' or negative['status']!='FAIL_EXECUTION'
                or negative.get('exit_code') != 134):
            args.output.write_text(json.dumps({'status':'UNQUALIFIED_SCORER','results':results},indent=2)+'\n')
            return 2
    for receipt in args.receipts:
        lines=receipt.read_text().splitlines()
        receipt_rows=[json.loads(line) for line in lines]
        if any(row.get('protocol')=='historical-c-bounded-repair-v3' for row in receipt_rows):
            from functorial_c_pilot import audit_repair_rows
            audit_repair_rows(receipt_rows,pair,{phase:{'code.c':source}
                              for phase,source in sources.items()})
        for line,row in zip(lines,receipt_rows):
            if row.get('kind')!='edit':
                continue
            if row['pair']!='fourier-horner' or row.get('protocol') not in (
                    'historical-c-pilot-v2','historical-c-bounded-repair-v3'):
                raise ValueError('Unsupported edit receipt protocol')
            phase=row['phase']
            if row['commit']!=pair[phase]['commit']:
                raise ValueError('Receipt source commit mismatch')
            item={k:row[k] for k in ('pair','phase','task_id','trial','commit')}
            item['protocol']=row['protocol']
            if row['protocol']=='historical-c-bounded-repair-v3':
                item['attempt']=row['attempt']
                item['model_response_sha256']=row.get('response_sha256')
                item['parent_response_sha256']=row.get('parent_response_sha256')
            item['response_sha256']=hashlib.sha256(line.encode()).hexdigest()
            if row['status']!='completed':
                item['result']={'status':'NOT_COMPLETED','semantic_correctness':None}
            else:
                try:
                    candidate=apply_patch(sources[phase],row['response']['message']['content'])
                    item['result']=evaluate(candidate,sources[phase],args.compiler,args.image)
                except ValueError as exc:
                    item['result']={'status':'FAIL_PATCH','semantic_correctness':None,'error':str(exc)}
            results.append(item)
    output={'status':'SCORER_QUALIFIED','compiler_sha256':hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
            'image':args.image,'driver_sha256':hashlib.sha256(DRIVER.encode()).hexdigest(),'results':results}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
