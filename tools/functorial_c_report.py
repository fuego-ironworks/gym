#!/usr/bin/env python3
"""Audit both downloaded pilot artifacts and emit an explicit paired report."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import functorial_c_eval as base
import functorial_c_pilot as pilot


def audit(bundle: Path, trials: int, offline: bool) -> dict:
    header_path = bundle / 'run.json'
    log_path = bundle / 'responses.jsonl'
    header = pilot.strict_json(header_path.read_text())
    rows = [pilot.strict_json(line) for line in log_path.read_text().splitlines()]
    manifest_hash = hashlib.sha256(base.MANIFEST.read_bytes()).hexdigest()
    if header['manifest_sha256'] != manifest_hash or header['protocol'] != pilot.PROTOCOL:
        raise ValueError('Manifest or protocol differs from retained run')
    pair = next(p for p in base.read_manifest()['pairs'] if p['id'] == bundle.name)
    sources = {phase: base.read_sources(pair, phase, base.ROOT / '.cache/functorial-c', offline)
               for phase in ('before', 'after')}
    tasks = {}
    for kind, field in (('question', 'questions'), ('edit', 'edits')):
        for original in pair.get(field, []):
            task = copy.deepcopy(original)
            if kind == 'edit':
                task['instruction'] = pilot.EDIT
            tasks[(kind, task['id'])] = task
    expected = {(kind, task_id, trial, phase) for kind, task_id in tasks
                for trial in range(trials) for phase in ('before', 'after')}
    seen = set()
    for row in rows:
        key = (row['kind'], row['task_id'], row['trial'], row['phase'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate response')
        seen.add(key)
        phase = row['phase']
        if (row['pair'] != pair['id'] or row['repository'] != pair['repository']
                or row['commit'] != pair[phase]['commit']
                or row['source_files'] != pair[phase]['files']
                or row['manifest_sha256'] != manifest_hash
                or row['model']['digest'] != header['model']['digest']):
            raise ValueError('Response provenance mismatch')
        task = tasks[(row['kind'], row['task_id'])]
        payload = row['request']
        wanted = [{'role': 'system', 'content': pilot.SYSTEM},
                  {'role': 'user', 'content': base.make_prompt(sources[phase], task, row['kind'])}]
        if payload['messages'] != wanted:
            raise ValueError('Prompt does not match the pinned source and task')
        settings = {**payload['options'], 'think': payload['think']}
        if settings != row['settings']:
            raise ValueError('Recorded settings do not match request')
        if 'response' in row:
            if pilot.outcome(row['response']) != row['status']:
                raise ValueError('Completion classification mismatch')
            if row['kind'] == 'question' and row['status'] == 'completed':
                actual = pilot.score_answer(row['response']['message']['content'],
                                            task['answer'], row['task_id'])
                if actual != row['score']:
                    raise ValueError('Stored question score cannot be reproduced')
    by_phase = {phase: {'planned': sum(key[3] == phase for key in expected),
                       'retained': sum(key[3] == phase for key in seen),
                       'completed_questions': 0, 'correct_questions': 0,
                       'question_statuses': {}, 'edit_statuses': {}}
                for phase in ('before', 'after')}
    task_rows = defaultdict(list)
    for row in rows:
        cell = by_phase[row['phase']]
        key = 'question_statuses' if row['kind'] == 'question' else 'edit_statuses'
        cell[key][row['status']] = cell[key].get(row['status'], 0) + 1
        if row['kind'] == 'question' and row['status'] == 'completed':
            cell['completed_questions'] += 1
            cell['correct_questions'] += int(row['score']['correct'])
        task_rows[(row['kind'], row['task_id'], row['phase'])].append(row)
    repeatability = []
    for (kind, task_id, phase), group in sorted(task_rows.items()):
        repeatability.append({'kind': kind, 'task_id': task_id, 'phase': phase,
                              'trials_retained': len(group),
                              'distinct_final_answers': len({
                                  r.get('response', {}).get('message', {}).get('content')
                                  for r in group if r['status'] == 'completed'})})
    return {'pair': pair['id'], 'model_digest': header['model']['digest'],
            'runtime': header['runtime'], 'source_commits': {
                phase: pair[phase]['commit'] for phase in ('before', 'after')},
            'planned_responses': len(expected), 'retained_responses': len(rows),
            'missing_responses': [list(k) for k in sorted(expected - seen)],
            'complete': expected == seen and all(r['status'] == 'completed' for r in rows),
            'paired': pilot.paired_summary(rows), 'by_phase': by_phase,
            'repeatability': repeatability,
            'receipt_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest(),
            'header_sha256': hashlib.sha256(header_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 3:
        parser.error('Expected 1..3 trials')
    result = {'protocol': pilot.PROTOCOL, 'pairs': [], 'errors': [],
              'interpretation': 'Historical versions confound structure, notation, comments and length. '
              'Repeated zero-temperature trials are not independent tasks. Edit applicability is not execution.'}
    try:
        for pair in base.read_manifest()['pairs']:
            matches = [p.parent for p in args.artifacts.rglob('run.json')
                       if p.parent.name == pair['id']]
            if len(matches) != 1:
                raise ValueError('Expected exactly one artifact bundle for ' + pair['id'])
            result['pairs'].append(audit(matches[0], args.trials, args.offline))
        if len({p['model_digest'] for p in result['pairs']}) != 1:
            raise ValueError('Different model weights across repository pairs')
        if len({json.dumps(p['runtime'], sort_keys=True) for p in result['pairs']}) != 1:
            raise ValueError('Different runtime versions across pairs')
        for log in args.artifacts.rglob('ollama.log'):
            if re.search(r'truncat\w*[^\n]{0,100}(?:input|prompt)|(?:input|prompt)[^\n]{0,100}truncat',
                         log.read_text(errors='replace'), re.I):
                raise ValueError('Potential input truncation requires review: ' + str(log))
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        result['errors'].append(str(exc))
    result['status'] = ('INVALID_OR_INCOMPLETE_EVIDENCE' if result['errors'] else
                        'COMPLETE' if all(p['complete'] for p in result['pairs']) else 'PARTIAL')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result['errors'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
