# Local model acceptance system

The intended workflow is automated qualification rather than manual model hobbyism.

## Inputs

There are two distinct input streams.

### Real-work acceptance cases

Cases should come from work that matters in practice, for example:

- repository and pull-request review;
- finding the smallest safe code change;
- shell scripting;
- compiler/backend work;
- mathematical reasoning;
- extracting and checking claims from documents;
- maintaining evidence boundaries;
- following repository policy such as `AGENTS.md`;
- completing multi-part requests without unnecessary prompting.

Each case should retain enough information to reproduce the evaluation: prompt/instructions, relevant context, expected properties, deterministic checks where available, and known-good or known-bad examples when useful.

### External claims

Blog posts, model cards, papers, benchmark reports, and leaderboard statements may be stored in machine-readable form. A useful record includes:

```yaml
source:
date:
models_compared: []
claimed_strength:
task_category:
hardware:
quantization:
prompting_conditions:
claimed_metric:
testable_hypotheses: []
```

An external claim is a hypothesis generator. It does not qualify a model by itself.

## Premature-stop handling

If a model routinely stops before completing a requested task, the normal response should be policy rather than human babysitting.

A continuation policy can conservatively detect signals such as:

- requested deliverables still absent;
- an unfinished numbered procedure;
- an unclosed generated structure;
- explicit statements that work remains;
- a task-specific completion check that still fails.

The wrapper may issue a continuation instruction and rerun completion checks. Every continuation must be counted and retained. Use a finite retry limit. The continuation policy must not create new scope merely because generation could continue.

Premature-stop rate and continuation count are evaluation measurements.

## Scoring dimensions

Keep dimensions separate instead of hiding them in a single score.

### Executable correctness

Compile, run tests, compare files, validate schemas, inspect exit status, or otherwise test produced artifacts directly.

### Repository behavior

Check whether the requested change occurred, whether unrelated changes appeared, whether policy was followed, and whether the model preserved the requested evidence boundary.

### Evidence behavior

Check whether claims are supported by the supplied sources and whether uncertainty or missing evidence remains represented as such.

### Instruction following

Represent explicit requirements as individually checkable properties where possible.

### Completion

Measure whether the requested deliverables exist, whether continuation was needed, and whether the model terminated with unresolved work.

### Conservatism

Record unnecessary edits, unsupported claims, invented implementation status, scope growth, and similar overreach.

### Resource cost

Record wall time, tokens, RAM/VRAM, and other machine costs when available and relevant.

### Open-ended judgment

Only use model judges where direct checks are inadequate. Prefer blinded comparison and retain individual judge outputs rather than treating the judge as ground truth.

## Qualification and routing

The useful output need not be one globally preferred model. The evidence may support routing such as:

```text
repository_review  -> model_a
code_generation    -> model_b
math               -> model_c
cheap_extraction   -> small_model_d
fallback           -> model_a
```

The routing table must be traceable to retained qualification results.

## Harness boundary

The repository may use existing evaluation frameworks underneath its adapters and scorers. The local acceptance-case format, evidence rules, continuation policy, qualification rules, and retained receipts remain authoritative here.

Suggested high-level layout:

```text
cases/
claims/
adapters/
scorers/
policies/
receipts/
docs/
```

The first implementation should stay small: one local inference adapter, a handful of representative real-work cases, deterministic scoring where possible, continuation accounting, and a receipt format. Broader benchmark integration can follow after that boundary works end to end.
