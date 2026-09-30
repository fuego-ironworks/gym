# LoRA run identity: first consumer hardening

Related: Gym #62, Gym PR #63 (Build Pythia 14M inspectability harness),
and isomorphisms/ai-ci#177.

## Boundary

This change hardens the existing Gym Python consumer. The existing Python
entrypoint remains migration debt; this patch creates no new Python entrypoint.
It does not implement an experiment-product scheduler or turn AICI into one.
AICI owns reusable acceptance contracts and negative fixtures. Shared provider
policy must not be duplicated in this model-specific consumer.

Paid execution is not implemented or authorized here. A plan and a passing
local/GitHub experiment are not spending authority. This patch does not protect
other provider scripts from direct invocation. Do not describe it as the full
paid-compute gate from #62 or ai-ci#177.

## Source identity

Resolve the requested Hugging Face revision **before** loading weights. Require
a full commit SHA, then give the same SHA to model and tokenizer loaders. Record
the original revision label separately. Never re-resolve a moving branch after
execution and call that result the identity of previously loaded weights.

Observe the local Git HEAD and compare it with `GYM_SOURCE_SHA` when supplied.
Reject tracked uncommitted changes. Both workflow jobs explicitly check out the
PR head rather than merely writing that head's name into a synthetic-merge
receipt. Untracked outputs are permitted; a clean-checkout test is not a sandbox
or a defense against hostile Python imports.

## Job identity and attempt identity

`plan.json` records the source commit, model repository and commit, target matrix,
rank, alpha, SGD learning rate, optimizer-step count, seed, example, device,
dtype, thread count, and selected environment versions. A SHA-256 over canonical
JSON identifies this job. Dictionary ordering and alternate revision labels for
the same model commit do not change it. Material parameter/source changes do.

This is a single-consumer identity, not yet the generic scientific-configuration
and execution-environment identity split that the cross-provider matrix needs.
A changed runtime currently means a different job. Package versions do not
close over transitive dependencies or binary hashes; environment closure remains
an open prerequisite for paid execution.

Each invocation creates an exclusive new directory below `--output-dir`, named
with the full job ID and GitHub run ID/attempt (or a local UUID). Reusing exactly
the same attempt fails rather than overwriting its evidence. A second GitHub
attempt uses a different directory. This preserves retries; it does **not**
resume interrupted optimizer state or deduplicate accepted computation.

## Plan before execution

Using the existing command, add `--plan-only` to resolve metadata and write
`plan.json` and `sources.json` without loading model weights or training.
Model-host metadata access still requires a connection. The output says
`PLANNED`, not `passed`, and does not contain `COMPLETE.json`.

Supply `--expected-plan PATH/plan.json` on an execution invocation to require
exact plan equality before weights are loaded. A changed model revision,
source commit, example, hyperparameter, or recorded runtime fails this check.
Missing/unreadable plans fail as well. This comparison is an integrity check,
not identity verification of an authorizer or an authenticated CI result.

The current loss remains the original harness's loss across all next-token
positions in the single example. `inputs.json` now preserves exact token IDs
and labels. This patch does not silently replace that objective with a
single-position loss or claim that such a test already ran.

## Completion and failure

A failed Test 0 stops execution before the optimizer runs. Each returned step
receipt must report success, and the number must equal the requested number.
The existing per-step tensor capture stays in `lora_differential.py`.

On successful execution the runner writes an index of the output files and
rereads them to check sizes and SHA-256 values. `COMPLETE.json` is written last
and binds the index's SHA-256. A plan alone, a partial `.pt` write, or a false
acceptance result cannot create this marker. A caught exception records
`FAILED.json` while retaining partial artifacts. A hard process kill can leave
neither final marker; classify that attempt as incomplete, never successful.

JSON metadata is written by temporary file plus atomic replacement. This is not
a claim of power-loss durability for the entire directory. Nor is a local hash
index a durable backup or an authenticated receipt. Artifact promotion,
read-back from the durable destination, and restore testing remain open.

The tests exercise the metadata boundary with fake model/Hugging Face classes,
real local filesystem operations, and a real temporary Git repository. They do
not establish updated Pythia numerical results, GPU parity, provider request
correctness, or a paid-runtime billing limit.

## Next shared contract

The full matrix requires bounded expansion with named axes and stable identities,
constraint checks before Cartesian expansion, separate scientific and execution
identity, and an explicit dependency graph for optimizer trajectories rather
than pretending each adjacent step is an independent trial.

A future provider-create boundary must consume externally authenticated smoke
and approval evidence bound to the exact plan, request, price inputs, deadline,
and spending limit. Store provider credentials only behind that boundary.
Self-written JSON saying `authorized: true` is not authorization. Runtime and
billing bounds require independently tested teardown and provider confirmation,
not merely a cost estimate or a workflow timeout.

