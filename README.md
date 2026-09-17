# gym

`gym` is the acceptance and qualification repository for local language models.

The goal is not to spend time manually comparing models. The repository turns real work into repeatable tests, checks model behavior against those tests, and retains receipts showing what each model actually established.

## Principles

- Prefer tests derived from real work over generic model reputation.
- Treat blog posts, model cards, and benchmark reports as claims to test, not conclusions to adopt.
- Keep machine-checkable results separate from model-graded judgments.
- Record premature stopping, unnecessary continuation, unsupported claims, and unnecessary changes as failures or costs rather than requiring manual babysitting.
- Preserve evidence boundaries: a model passing one task does not establish broader capability.
- Keep retained receipts plain and durable; terminal presentation may use color, but decoration is not evidence.
- Promote or route models by task category when that is better supported than declaring one model globally best.

## Intended areas

- `cases/` — real acceptance cases, grouped by kind of work.
- `claims/` — machine-readable external claims about models and benchmarks.
- `adapters/` — local inference backends and model runners.
- `scorers/` — executable, structural, evidence, completion, and conservative-behavior checks.
- `policies/` — continuation, retry, fallback, and routing policy.
- `receipts/` — retained qualification results.
- `docs/` — design notes and evidence rules.

The first target is a thin harness that can compare local models without requiring repeated manual prompting or subjective model tinkering.
