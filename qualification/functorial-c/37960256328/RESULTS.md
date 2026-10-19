# GPT-OSS on real historical C: completed pilot

## Results

The actual `gpt-oss:20b` model completed all 72 planned responses: 66 answers
and six edit attempts. Both historical source versions received the same tasks
and settings, in separate conversations, with three trials each.

| Task | Ordinary C, before | Functorial/Icky C, after |
|---|---:|---:|
| Fourier polynomial questions: five questions, three trials | 15/15 correct | 15/15 correct |
| Seifert questions: six questions, three trials | 18/18 correct | 18/18 correct |
| Delivered unified patch: one edit, three trials | 0/3 valid | 0/3 valid |

All 66 question answers were correct. This is 11 distinct questions, not 66
independent examples. All 72 requests ended normally with `done_reason=stop`;
none of the failures below was an output-truncation failure.

This pilot supplies **no question-accuracy advantage for either style**. The
questions are small and likely too easy to distinguish stronger comprehension.
A tie here neither establishes universal equivalence nor supports the original
claim that Functorial C is easier for ChatGPT to read.

## What happened on the edit task

The request was to accept null output without reading input or writing output,
remove the `static 2` minimum-array-bound contract, keep the existing ABI and
header-free source, and preserve all non-null behavior and the source dialect.

Every response used a bare `@@` instead of a valid unified hunk header with line
ranges. The responses also included a nonstandard `*** End of File ***` trailer.
Both Git's applicability check and the strict single-file checker rejected all
six. **No model patch reached compilation or runtime execution.** The delivered
edit score is therefore 0/3 before and 0/3 after, not a successful code repair.

Inspection of the proposed changes reveals an additional distinction:

- All three ordinary-C responses proposed changing `output_cartesian[static 2]`
  to `output_cartesian[2]` and inserting `if (output_cartesian == 0) return;`
  before the existing input processing. The proposed changes address the two
  requested requirements, but the malformed patches were not repaired or
  promoted into executable successes.
- All three Functorial/Icky-C responses retained the original exported function
  and appended another definition of `fourier_polynomial_cartesian_ick`.
  Both definitions still used `output_cartesian[static 2]`. Thus these responses
  had substantive code/requirement errors in addition to malformed patch format.

This is evidence of a difference on **this one edit request**, not a general
conclusion that Functorial C is harder. Patch-format failure must remain distinct
from source understanding; no model self-rating was used.

## Executable scorer qualification

The independent checker built actual ICK from source and compiled the original
Icky text without glyph normalization or a replacement source compiler.
Positive controls passed for both historical versions. Each passed 882 sampled
historical comparisons plus independent polynomial, sentinel and null-output
checks. Deliberately corrupted output-store controls failed by executed assertions
with exit status 134 for both versions, rather than failing to compile.

Those controls qualify the checker; they are **not model-generated solutions**.
The workflow's success means the evaluation ran and retained its results. It does
not mean the six model patches passed. The retained `model-edits.json` classifies
all six as `FAIL_PATCH` before compilation. Its legacy `semantic_correctness:false`
field on that status must not be read as an executed semantic test: no candidate
program was built. The stage/status is the controlling evidence boundary.

## Reproducibility and independently checked provenance

- Model: `gpt-oss:20b`, reported 20.9B parameters, MXFP4 GGUF/llamacpp.
- Model digest, identical in both jobs: `f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe`.
- Runtime: Ollama `0.40.2`.
- Protocol: `historical-c-pilot-v2`; temperature 0; low reasoning effort;
  matched seeds; 16,384 context tokens.
- Manifest SHA-256: `752f54828b266c4b790243dd01adb2710273f9d06041a19108ca13f4a4855250`.
- Model-run experiment commit: `0afc6f4d7234076d25ada7b4713e3d3de0bf81f7`.
- Executable-check experiment commit: `577d2f5f986d0a5b3f300853a97ba44334bbfd34`.
- ICK source: `c61e448251744a2f40ad743ebef1a027bdcd2f9d`.
- ICK compiler binary SHA-256: `45805e3295f2709737fcc87985a8a2341aa8663ff8e4e398d2a9d78dc1283e10`.
- Native runtime image: `sha256:8c469c7c543c2a452edf9140b7e4ccf9a1b80511a9b2b38892cf73da5ede167e`.

The downloaded archives matched their GitHub artifact SHA-256 digests.
An independent audit extracted the source text from every retained model request,
recomputed Git blob hashes against the pinned historical blobs, checked source
commits/model identities/matched settings, and recomputed all 66 question scores.
All 72 server slot-release records reported `truncated = 0`.

The three zero-temperature trials measure repeatability, not independent tasks.
Original comments remain visible. Structure, names, glyphs, comments and length
change together; their causal effects are not isolated. This tests GPT-OSS, not
GPT-6, human readability, or the entire repository portfolio.

## Retained raw artifacts

Model run: https://github.com/fuego-ironworks/gym/actions/runs/37960256328

Executable-check run: https://github.com/fuego-ironworks/gym/actions/runs/37960963198

| Artifact | GitHub artifact ID | ZIP SHA-256 |
|---|---:|---|
| Fourier prompts, responses and logs | 11632676951 | 15992be297577680ca372e594f08df1bbf609b9f647d50d3446f6d69b98f5b89 |
| Seifert prompts, responses and logs | 11632291247 | c1e32e118eb7a993e49efc3508e731772b17a00f6be298bf279a28f1a094b920 |
| ICK controls and patch outcomes | 11632990822 | 73125080095253b933e51b9eb48bfa1ccfb0f3060a329a8f1ea75fc092aee228 |

Fourier `responses.jsonl` SHA-256:
`74f90cc7479a169cf28e23b9ca100b879b63ca374061394d8c9fef7abc61a519`.

Seifert `responses.jsonl` SHA-256:
`114f0d8df131724c60be83c84be81ea7dbd8eb06f2495e8d34ebe27db841d33f`.

The hosted artifacts expire November 8, 2026. A separate archive of all three
ZIPs and the independent audit was created with this report; retain it for replay.
No model weights, original application changes or private conversation history
are committed here.
