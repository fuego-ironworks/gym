# GPT-OSS bounded repair v3 — no valid delivered patch

Date: 2026-10-09.

**One explicit format-feedback turn produced no applicable patch.** All twelve
requested model responses were retained. Eleven ended normally; one Icky repair
reached the 1,536-token output limit. The run is incomplete under its declared
completion rule, and no v3 candidate reached compilation or execution.

Model run: https://github.com/fuego-ironworks/gym/actions/runs/37993849150

Experiment commit: `7104d58c1765c4b544d81a99d41272feabfcfb79`.

## Observed results

| Measure | Ordinary C, before | Functorial/Icky C, after |
|---|---:|---:|
| Fresh first responses retained | 3 | 3 |
| First responses ending normally | 3 | 3 |
| First responses rejected for patch syntax | 3 | 3 |
| Applicable first patches | 0 | 0 |
| Repair responses retained | 3 | 3 |
| Repair responses ending normally | 3 | 2 |
| Completed repairs rejected for patch syntax | 3 | 2 |
| Repairs stopped by the output limit | 0 | 1 |
| Applicable repaired patches | 0 | 0 |
| Candidate compilation / execution | NOT_RUN | NOT_RUN |

The output-limited repair is an incomplete response, not a completed wrong answer.
All six first attempts and the five normally completed repairs produced the
strict parser diagnostic `Expected a unified hunk`. Git's independent
applicability check also rejected those eleven responses with
`No valid patches in input`. No model text was repaired by the evaluator.

The model step ran from 21:33:10 to 21:54:46 UTC. Recorded request durations total
1,294.69 seconds; the longest request took 249.05 seconds. Neither the
35-minute experiment budget nor a request timeout caused this failure.
After retaining all twelve responses, the full-completion audit raised
`Incomplete bounded repair run`, and the job exited with status 2.

## What the model actually returned

The feedback repeated the actual parser rejection and specified decimal line
ranges, exact hunk counts, context/removal/addition prefixes and no extra
trailers. It supplied no gold patch or semantic correction.

- **Ordinary C, all three first attempts and all three repairs:** the proposals
  changed `output_cartesian[static 2]` to `output_cartesian[2]` and placed an
  `output_cartesian == 0` guard before existing input processing. They still
  emitted a bare `@@` and the forbidden `*** End of File ***` trailer.
- **Icky C, all three first attempts:** the responses retained the original
  exported function and appended a second definition of the same symbol. Both
  definitions retained `static 2`. They also used the malformed hunk format.
- **Icky C, the two normally completed repairs:** the visible proposals now
  replaced the exported function, removed `static 2` and placed the null guard
  before coefficient access. However, they inserted literal source line numbers
  such as `50 void ...` into the removed and added C text while retaining a bare
  `@@` and the nonstandard trailer. Those numbered lines do not match the pinned
  source, and the additions are not valid C source as delivered.
- **Icky C, the output-limited repair (trial 1):** the response began a numeric
  hunk header but stopped inside the deletion portion. It had
  `done_reason=length` and `eval_count=1536`. The unfinished proposal cannot
  establish compliance with the complete edit request.

This is inspection of visible proposals, not execution of reconstructed files.
The completed proposals introduced no include directives. The visible Icky
operations kept their `←` assignments; no dialect normalization was performed.
The intended signature changes above preserve the exported symbol and pointer
ABI at the proposal level, but **zero applicable candidate files exist**.
Header-free appearance, preserved glyphs and an apparently appropriate guard do
not convert malformed output into accepted source or sampled runtime correctness.

## Exact experiment and replay

Protocol: `historical-c-bounded-repair-v3`.

The first prompts, edit request, historical source bytes, model digest, runtime
and generation settings match v2. Only the six Fourier edits were requested;
the 66 already saturated question calls were not repeated.

- Model: `gpt-oss:20b`, reported 20.9B parameters, MXFP4 GGUF/llamacpp.
- Model digest:
  `f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe`.
- Runtime: Ollama `0.40.2`; Python `3.12.3`.
- Generation: temperature 0, low reasoning effort, context 16,384 tokens,
  output allowance 1,536 tokens, four threads, seeds 20261009–20261011.
- Host: Ubuntu runner, Linux x86_64, four vCPUs on AMD EPYC 7763;
  recorded RAM 16,766,414,848 bytes.
- Manifest SHA-256:
  `752f54828b266c4b790243dd01adb2710273f9d06041a19108ca13f4a4855250`.
- Pilot SHA-256:
  `0aea541a4073f692e6338c1ca25d88f97b521fea11f64aba4f1e0569bbe5b1c0`.
- Semantic-checker source SHA-256:
  `92ebc44bd17d028eda146f9366cbd6be1c1e17f92b0037b31542944da353c72b`.
  This identifies source only; the v3 ICK execution path did not run.

The original `isomorphismes/Fourier-sound` snapshots were independently fetched
and their Git blob hashes verified:

| Version | Source commit | Git blob |
|---|---|---|
| Before | `ac0e9b40a556012b42be240f1c32c88370028b4a` | `a0a544022bd6b8ee12ea36d204f2c6d70b7047a6` |
| After | `e5d270bd6e53f111d22b51111b2804024fb48878` | `340661cc3bc02607337f59ccaa16949d46fec942` |

Source path in both versions: `fourier/ick_polynomial_leaf.c`.

Offline replay with `require_complete=False` verified the exact source-bearing
requests, unchanged settings, response hashes, one-repair parent links and
all eleven completed patch scores. It returned `complete=false`.
Replay with `require_complete=True` correctly rejected the run as incomplete.
The original completion/status fields were not changed.

All twelve server slot-release records contain `truncated = 0`. That log field
does not override the one API response with `done_reason=length`; the latter
remains classified as output-truncated.

## Controls and execution boundary

V3 ICK build, positive controls, negative controls, candidate compilation and
candidate execution are **NOT_RUN**. The failed inference step prevented those
later steps, and no response had passed the patch parser.

The earlier v2 controls remain historical evidence: both positive controls
passed 882 comparisons plus independent/sentinel/null checks, and both deliberately
broken controls failed executed assertions with exit 134 in
[run 37960963198](https://github.com/fuego-ironworks/gym/actions/runs/37960963198).
They are not new v3 qualification. No extra compiler or model run was launched
to replace this failure with a green workflow.

## Retained artifacts and hashes

Artifact: `gpt-oss-repair-v3-fourier-horner-7104d58c1765c4b544d81a99d41272feabfcfb79`.

GitHub artifact ID: `11647171983`. Expiration: November 8, 2026.

| Retained object | SHA-256 |
|---|---|
| Original artifact ZIP | `4c5e30ed13e335759ff6f8108081b3770acc662bb80dd088c60ecef55d76cae9` |
| `responses.jsonl` | `d80c1550d479bc01b69dd3b8a59e6ca511638f2745db67047573d3d957441893` |
| `run.json` | `7bd0f0c92febdf20cc68135b78f5721016e1434f25eca7ba82689687d756ce07` |
| `blocked.json` | `3142f818732374fd67fec255a57828069fcc609d5364b42dd3cd21e6732bf943` |

The ZIP SHA-256 matches GitHub's artifact metadata, and all ten extracted members
match the original archive bytes. The local evidence bundle retains the original
ZIP, unmodified model receipts, logs, source snapshots, API artifact metadata,
and the separate partial/complete replay audit.

## Comparison with unchanged v2

[V2's results](../37960256328/RESULTS.md) remain unchanged: all 66 question answers
were correct across eleven distinct questions, while all six delivered edit
patches were malformed. V3 reproduced the initial edit failure pattern and
tested one bounded format-feedback procedure. That procedure produced no usable
patch on this task.

The two completed Icky repairs visibly address source requirements they missed
in their first attempts, but still fail delivery. This distinction supports
keeping source reasoning, patch formatting, output completion and executable
behavior separate in future evaluation.

This small within-model experiment does not establish a general readability
ranking. Three temperature-zero trials do not supply three independent tasks.
The historical versions change structure, names, comments, length and notation
together; v3 also uses a different physical CPU model from the earlier run.
The retained failure warrants reconsidering this particular feedback procedure,
not silently relaxing its acceptance rules or claiming that no other procedure
could succeed.
