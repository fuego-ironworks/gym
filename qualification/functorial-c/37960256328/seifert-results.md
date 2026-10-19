# Seifert: actual GPT-OSS pilot results

Run: https://github.com/fuego-ironworks/gym/actions/runs/37960256328
Job: `113920991495`; artifact: `11632291247`.
Experiment source: `0afc6f4d7234076d25ada7b4713e3d3de0bf81f7`.
Protocol: `historical-c-pilot-v2`.

## Result

All six distinct questions were answered correctly on both historical versions
in all three trials. Before: 18/18; after: 18/18. All 36 responses completed with
`done_reason=stop`. No before-only or after-only successful pairs occurred.
The three temperature-zero trials measure repeatability, not independent tasks.

| Question | Answer | Before | After |
|---|---|---:|---:|
| short-ribbon-vertices | 12 | 3/3 | 3/3 |
| max-ribbon-indices | 6144 | 3/3 | 3/3 |
| corner-center | x=1.9, y=1.9, z=0 | 3/3 | 3/3 |
| invalid-large-turn | SEIFERT_INVALID_ARGUMENT | 3/3 | 3/3 |
| ribbon-width-invalid | false | 3/3 | 3/3 |
| invalid-segments | 0 | 3/3 | 3/3 |

This is a tie on these questions. It is not evidence that Functorial C is easier,
or proof that the styles are equally readable on more difficult tasks. The source
versions differ in structure, length, names, notation and comments. These results
concern GPT-OSS, not an empirical measurement of GPT-6 comprehension.

## Model and source identities

- Model: `gpt-oss:20b`, reported 20.9B parameters, MXFP4, GGUF/llamacpp.
- Model digest: `f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe`.
- Runtime: Ollama `0.40.2`.
- Before: `eb1824cfb7acfd8344a03a0e399a6241e03f8724`.
- After: `d7aae04d12a25364bf9348166a2b6b0b9cb15972`.
- Manifest SHA-256: `752f54828b266c4b790243dd01adb2710273f9d06041a19108ca13f4a4855250`.

Prompt token counts ranged from 3,884 to 5,005, below the requested 16,384-token
context. All 36 server slot-release log entries report `truncated = 0`.

## Retained evidence

The artifact includes exact full prompts, original source text, raw model responses
and thinking fields, model/runtime metadata, request settings, timing, and server logs.
The downloaded ZIP hash matched GitHub's artifact digest:

- ZIP SHA-256: `c1e32e118eb7a993e49efc3508e731772b17a00f6be298bf279a28f1a094b920`.
- `responses.jsonl` SHA-256: `114f0d8df131724c60be83c84be81ea7dbd8eb06f2495e8d34ebe27db841d33f`.
- `run.json` SHA-256: `7cfc230353c94363cfcc33285a849edba051fbfa7f768af80bdc54713d4f68ee`.

GitHub's artifact expires on November 8, 2026. This committed summary is durable;
the complete raw evidence also needs an archived copy of the ZIP for long-term replay.
The Fourier and executable-patch results are separate, not implied by this result.
