# Experiment contract v1: offline core, paid execution disabled

Related: [Gym #62](https://github.com/fuego-ironworks/gym/issues/62),
[AICI #177](https://github.com/isomorphisms/ai-ci/issues/177), and
[14M harness #63](https://github.com/fuego-ironworks/gym/pull/63).

This change implements deterministic product expansion, an independent artifact
receipt reader, and an offline paid-preflight contract checker. It does **not**
implement a scheduler, trusted approval service, provider client, automatic
retirement, adapter resume, or durable storage promotion. Existing Vast scripts
are not retroactively guarded by this module. Keep paid execution disabled.

## Identities and nested products

A `case_id` is SHA-256 of a domain-separated, canonical JSON case: exact source
repository/commit, clean-source assertion, environment-manifest digest,
input-manifest digest, required check names, and all scientific coordinates.
A `job_id` also binds provider, resource envelope and the complete rendered
request. Retrying the same job keeps its identity; run/attempt IDs belong to
execution evidence. Changing machine or provider keeps the case but changes
its job. Never reuse the old job's approval after that change.

Matrices contain `schema: 1`, `source`, `environment_sha256`, `input_sha256`,
`required_checks`, `fixed`, `axes`, `exclude`, and `max_cases`. Dotted axis names
preserve nested coordinates, e.g. `adapter.target`. A choice can be a whole
object: make each model repository/revision pair one choice, not independently
crossed axes. Optional `coordinates.model` must contain `repository` and an
immutable `revision`. Resolve names such as `step143000` to commit IDs before
expansion; no remote resolution occurs here.

The tests' `spec()` is an executable **synthetic fixture**, not a real pinned
Pythia run. It expands two targets and two seeds while keeping one example,
rank one and two adjacent optimizer updates fixed. Optimizer steps within one
trajectory are dependent observations, not four independent jobs. A producer
must still retain every adjacent step, as required by #63.

Key/axis/value ordering does not change IDs. Exact duplicate choices collapse.
Booleans and integers stay distinct. Floating-point JSON values are rejected;
use explicit decimal strings for learning rates or money in integer microunits.
Equivalent decimal spellings are not normalized in v1: `"0.1"` and `"0.10"`
are deliberately different representations. Do not depend on approximate
numerical equality to deduplicate experiments.

Overlapping axes, overriding fixed values, empty axes, unknown exclusion
coordinates, exclusions that match nothing, and an empty final product fail.
The product limit is checked before exclusions/materialization: at most
`max_cases`, with an absolute ceiling of 4096. Large plans must be partitioned
explicitly; no truncation or automatic expenditure is allowed. A digest
collision between different cases fails instead of dropping a case.

## Ordered preflight contract

The required trace is:

```
source-verify -> compatibility -> free-smoke -> resource-estimate
              -> provider-render -> paid-authorize
```

Every event binds case ID, job ID, source commit, timestamp and an evidence
payload digest. Missing/reordered events and every status other than `pass`
fail. Evidence must be ordered, not future-dated and no more than one hour old.
Source identity, required check coverage, environment identity and smoke
run/attempt are validated. Skipped/cancelled/unknown checks are not successes.
A surrogate smoke is not exact coverage. GPU preflight is explicitly unsupported
and rejected in v1; a CPU smoke cannot qualify a GPU rental.

Resource evidence must be identified as measured and fit the requested RAM,
disk and runtime limits. Provider rendering must report zero mutations and
match the exact request digest. Price and approval use the same currency and
integer microunits. Quotes and approvals expire. Approval binds the complete
preceding trace, not just a job name, and its maximum must cover the quote.
Changing even a cheaper quote invalidates the previous approval binding.

**Trust boundary:** these checks prove consistency of supplied documents, not
that a measurement, GitHub result, operator identity or permission is authentic.
A hash is not a signature. The test approval is synthetic and never authority.
Even a successful audit always returns `execution_enabled: false`. Exit status
zero means only that the offline contract is valid. Never use it to authorize
a provider call. No network/process API or provider creation command exists.

AICI's existing `tests/build_preflight_oracle.c` still supplies source/build
ordering policy. This Gym contract supplies the experiment-domain predicates
for the extension tracked in #177; it does not replace that oracle or turn
AICI into a provider scheduler. The live boundary must obtain trusted CI
results, authenticate operator approval independently of agent-written files,
consume single-use authorization durably, reserve aggregate budget atomically,
and enforce timeouts/cleanup outside the rented process. A cost estimate or a
maximum written in JSON is not a hard billing cap.

## Run receipts and read-back

A bundle contains `run.json` with schema 1, case/job IDs, status `complete`, an
exact map of required checks to `pass`, and a nonempty artifact list. Each
artifact records `path`, `size_bytes` and SHA-256. Use separate job/run/attempt
directories. Write payloads first and publish the completed receipt last.

`verify-run` independently reads and hashes every payload. It rejects missing,
truncated or changed bytes; unknown extra files; unsafe paths; duplicate paths;
symlinks; nonregular files; incomplete runs; missing checks and wrong identities.
It never unpickles tensor files. Persist its receipt digest in a trusted external
index if authenticity rather than local consistency is required. Verification
assumes a quiescent operator-controlled directory, not a hostile concurrent
writer. Do not delete local evidence based on this check alone.

Interrupted-publication fixtures prove that payloads without the final receipt
are not a completed run. They do not prove model/optimizer recovery. Existing
#63 output needs a producer integration before it satisfies this format.

## Local and CI execution

```
python3 -m unittest discover -s tests -p test_experiment_contract.py -v
python3 tools/experiment_contract.py expand matrix.json > expanded.json
python3 tools/experiment_contract.py audit case.json job.json events.json
python3 tools/experiment_contract.py verify-run case.json job.json RUN_DIRECTORY
```

The dedicated GitHub workflow uses a standard Ubuntu runner, read-only repository
permissions, pinned actions, a five-minute timeout and no provider credentials.
It executes the same standard-library suite and retains its test log. No model,
third-party dependency, GPU rental or billable provider call is needed.

## Remaining acceptance before any paid run

Wire this contract into the actual BME/LoRA producers and an independently
trusted lifecycle gate, rather than accepting manually fabricated traces.
Resolve model/tokenizer commits **before loading**, not after execution. Finish
immutable environment acquisition, resource calibration, kill/resume testing,
collision-safe output publication, aggregate budget reservation, provider-side
cleanup verification and durable artifact promotion/read-back. Keep #62 and
#177 open until those integrations have executable evidence.
