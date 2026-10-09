# Historical C / GPT-OSS pilot v2

This extends draft PR #66. Production source is still taken, byte for byte,
from the two real historical pairs in `cases/functorial_c/pairs.json`.
No new application codebase is supplied to the model.

## Actual execution entrypoint

Use `tools/functorial_c_pilot.py` for the bounded experiment. The earlier
`functorial_c_eval.py` remains the shared source/prompt machinery and a legacy
exploratory runner; its patch-only scores are not semantic correctness scores.

The v2 pilot asks 11 fixed-answer questions and one edit request, on both
historical versions, for three trials: 66 answer calls and six patch calls.
The two repository pairs run independently. Every request is a new conversation
with only a system instruction and its code/question. The order is shuffled.
Source comments are retained, so wrapper-label blinding is not complete blinding.

The checked-in workflow uses public Ubuntu CPU runners, not a paid inference API.
It downloads the real `gpt-oss:20b` weights into the temporary runner, verifies
Ollama 0.40.2's published archive SHA-256, records the installed model digest,
and rejects changes to that digest within a run. Model binaries are not committed.
The ephemeral runner is not a persistent model hosting service.

The initial execution is run `37960256328`, source head
`0afc6f4d7234076d25ada7b4713e3d3de0bf81f7`. Its progress or a green harness
must never be reported as a model score. Only retained model responses count.

To reproduce on an existing Ollama host, first start its service and pull the
model. Then, from the Gym root, for example:

```sh
python3 tools/functorial_c_pilot.py --pair fourier-horner --trials 3 \
  --expected-digest YOUR_RETAINED_MODEL_DIGEST \
  --output runs/functorial-c/fourier-new-run
python3 tools/functorial_c_pilot.py --pair seifert-ribbons --trials 3 \
  --expected-digest YOUR_RETAINED_MODEL_DIGEST \
  --output runs/functorial-c/seifert-new-run
```

Output directories must not already exist; this prevents overwriting old evidence.
The model tag is not an immutable identifier: use the recorded digest to detect
an accidentally different model. The model/template metadata and runtime version
are retained in `run.json`; the exact full requests, responses, thinking fields,
completion reasons and timings are retained in `responses.jsonl`.

## Fixed controls

Both variants use temperature 0, the same trial seed, low reasoning effort,
16,384 context tokens, and a finite output allowance: 768 tokens for questions
or 1,536 for patches. Three zero-temperature trials measure repeatability;
they do not turn 11 questions into 33 independent questions.

V2 rejects duplicate JSON keys and non-JSON constants, distinguishes booleans
from numbers, and permits 1e-6 absolute error only for the Seifert center question.
That question can reasonably be answered using either decimal source literals
or their stored float values. Other answers retain exact typed comparison.

Incomplete, empty, truncated and infrastructure-failed responses are retained
separately. Missing partners are incomplete pairs, not evidence that one style
won. The paired summary rejects duplicate trial/phase receipts, different model
digests and mismatched sampling settings.

The job has a finite run budget. It stops on an infrastructure failure rather
than generating repeated misleading failures. A missing or partial final summary
must not be interpreted as a complete run.

## Corrected edit request

The original request overlooked that the production function declares its
output as `double output_cartesian[static 2]`. Allowing null output requires
removing that minimum-array-bound contract as well as returning before reading
input or output. The v2 request states both changes explicitly, preserves the
exported ABI and header-free source, and requires unchanged non-null behavior.
This revised request is retained verbatim in each model request; the old manifest
is not silently presented as the exact v2 edit prompt.

## Executable checking

`tools/functorial_c_semantics.py` verifies text-only, single-file patches against
the original source, with exact hunk ranges and context. It rejects additional
files, Git binary/mode directives, stale context and no-op patches.

The executable workflow builds ICK from the existing source-pinned AI-CI producer.
It compiles original Icky source directly, without glyph normalization or an
automatic switch to another compiler. GCC 13 packages in the container supply
native headers and link runtime; they are not the selected source compiler.

Model candidates execute in a Docker container with no network, a read-only
root filesystem, dropped capabilities, no new privileges, a non-root UID,
restricted process/memory/CPU resources, and a wall timeout. Only a disposable
work directory is writable. Compiler stage and resolved image identity are
retained with the checker results.

The checker requires positive controls to pass and an intentionally corrupted
output store to fail by an executed assertion, not a compiler failure. For each
candidate it checks 882 historical comparisons spanning count clamping, empty
inputs, signed zeros, nonfinite points and input/output aliasing; it also checks
output sentinels, an independent 2+5i result, and null-output early return.
These are sampled regression checks, not a proof of universal equivalence.

Control results are explicitly labeled as controls. They are not GPT-OSS answers.
Patch applicability, compilation and execution must remain distinct in reports.
The original source repositories are not changed by this experiment.

## Interpretation

This is a small, exploratory within-model comparison of complete historical
source versions. The code versions differ in structure, names, length, comments
and Unicode notation. No result here isolates one of those factors, establishes
human readability, or establishes how a different GPT model reads the code.

A tie on simple questions should be reported as a tie, not as support for the
initial claim that Functorial C is easier. More demanding questions and genuine
maintenance tasks can be added as a separately versioned experiment after this
pilot, without rewriting these answers or historical source snapshots.
