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

## Durable schemas

The first executable survivability slice lives in `schemas/`, with a standard-library-only validator in `tools/validate_record.py`. It defines provider-neutral model manifests, run records, and durable receipts while leaving private trace payloads and large model objects outside this public repository.

Run its local checks with:

```sh
python3 -m unittest discover -s tests
```

## Local object storage

`tools/object_store.py` stores large or private retained objects outside Git
under paths derived from their SHA-256 digest and verifies retained bytes before
they are trusted. It never treats content addressing as permission to publish
restricted objects.

See `docs/object-store.md` for the storage layout, commands, and exact evidence
boundary.

## Local Git mirrors

`tools/mirror_git.py` creates and updates complete bare Git mirrors from a
provider-neutral manifest, verifies mirror refspec and object integrity, and can
re-check retained mirrors without contacting the original host.

See `docs/git-mirroring.md` for manifest format, local scheduling, receipts,
and the boundary between Git history and hosted metadata.

## GitHub metadata snapshots

`tools/export_github_metadata.py` creates immutable local snapshots of
explicitly selected issue, pull-request, release, and workflow-run metadata.
Every retained JSON file is bound into the snapshot by SHA-256.

See `docs/github-metadata-export.md` for selection format and the separate
boundary for workflow logs, artifact bytes, and release assets.
