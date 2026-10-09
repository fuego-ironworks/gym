# Historical C / GPT-OSS bounded repair v3

This follow-up asks whether one concrete patch-checker rejection helps GPT-OSS
deliver the edit it failed to deliver in the first pilot. The earlier 72 model
responses, source snapshots, prompts and scores remain unchanged in
[the v2 results](../qualification/functorial-c/37960256328/RESULTS.md).

All six v2 patches had bare `@@` hunk headers and a nonstandard end-of-file
trailer. The three ordinary-C proposals also contained the requested guard and
array-contract change; the three Icky proposals duplicated the exported function
and kept `static 2`. These are different failure classes. A format-only retry
must not be reported as successful source editing unless its result also passes
the existing ICK checks.

## Fixed task and model

Protocol identifier: `historical-c-bounded-repair-v3`.

- Use only the `fourier-horner` pair from `cases/functorial_c/pairs.json`, with
  complete source bytes checked against the original Git blob hashes.
- Keep the exact v2 edit request, system instruction and initial user prompt.
  No newly invented application code is given to the model.
- Run three trials on each historical version: six fresh first responses.
  The saturated 66 question calls are not repeated.
- Require `gpt-oss:20b` digest
  `f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe`
  and Ollama `0.40.2`. A changed model tag or runtime blocks the experiment.
- Keep temperature zero, low reasoning effort, 16,384 context tokens,
  1,536 output tokens, four CPU threads, and seeds 20261009 through 20261011.
  These repeated trials measure repeatability, not independent tasks.

The manifest SHA-256 is pinned to the v2 manifest. V3 does not silently revise
the baseline source pair or the original task to make it easier.

## One bounded feedback turn

The first request is a fresh conversation. The strict single-file patch parser
then examines the exact response. If the response completed normally and the
parser rejects it, the harness appends the model's unmodified answer and one
feedback message to that conversation. The feedback includes only the parser's
actual rejection and the required unified-diff grammar: decimal hunk ranges,
exact line counts, context/removal/addition prefixes, and no extra trailers.

The feedback supplies no gold patch, expected numerical result, semantic fix,
or hint about which source edit to make. The model must reissue a complete patch
against the unchanged original file. Its second response is final even if it
is still rejected. An applicable first response receives no unnecessary retry.
Thus the experiment makes six to twelve model requests, with no third turn.

No response text is repaired mechanically. Both first and repair responses are
retained in `responses.jsonl`, together with the complete requests, settings,
source provenance, hashes and completion reasons. Each repair names the hash
of the first response it follows. Receipt replay reconstructs the exact prompt
and feedback, recomputes the patch score, rejects changed model/settings/source
identity and rejects missing, duplicate or extra repair turns.

An infrastructure failure stops further requests and preserves the partial run.
Empty, incomplete and truncated replies are not parser failures eligible for a
format retry. A finite 35-minute inference budget and per-request timeouts bound
the run. Output directories must be new so prior evidence cannot be overwritten.

## Hosted execution and semantic checking

The existing `.github/workflows/gpt-oss-pilot.yml` accepts mode
`bounded-repair-v3`. The branch push marker `[run-gpt-oss-repair]` selects the
same mode. The older `[run-gpt-oss]` marker and explicit `pilot-v2` mode keep the
v2 path available. A v3 run selects only the Fourier pair.

After inference, the workflow stops Ollama and releases model memory before
building the same source-pinned ICK compiler used by v2. It reuses the existing
networkless, non-root, resource-limited container checker. Positive and executed
negative controls must qualify the checker before any candidate receives a
semantic result. First and repaired responses are checked independently;
both outcomes are retained, even when the repair regresses.

Compilation and execution use separate invocations of the same restricted
container, with one shared 60-second candidate budget. Compiler failures are
`FAIL_COMPILE` with no executed semantic result. Only a compiled candidate's
execution can produce `PASS` or `FAIL_EXECUTION`. Both stage exits and diagnostics
are retained; a source-contract rejection also has no executed semantic result.

The model receipt artifact is named `gpt-oss-repair-v3-fourier-horner-<commit>`.
The separate compiler/control/candidate artifact is
`gpt-oss-repair-v3-executable-<commit>`. These names distinguish v3 receipts from
the frozen v2 artifacts. Inference receipts survive a later compiler build or
execution failure because they are uploaded before ICK qualification.

The runner and semantic checker extend Gym's existing Python integration;
the hosted workflow retains its existing installation boundary. This experiment
does not replace the model, compiler, source dialect or evaluation mechanism.

## Reading the results

Report four separate counts for each source version: applicable first patches,
applicable repaired patches, first candidates passing ICK checks, and repaired
candidates passing ICK checks. A patch rejection has no executed semantic result.
A green workflow means the evaluation completed; its model candidates can still
all fail. Compiler/control failure leaves candidate semantics unavailable.

V3 measures the model plus one explicit format-feedback procedure on this one
maintenance task. It does not retroactively improve v2 scores, establish a
general readability ranking, or isolate the effects of notation, names,
structure and source length. A successful repair would support trying this
bounded continuation policy on further real work; it would not justify silently
relaxing the acceptance rules.
