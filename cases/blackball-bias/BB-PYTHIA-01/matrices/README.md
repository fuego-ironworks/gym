# Explicit matrix batches

The four fixture-shard files each expand 48 exact authored prompts. The six-case BME replay plan crosses three retained candidate pairs with two recorded immutable model commits. Both lanes passed the existing offline expansion contract. They produced 198 unique case IDs; no model ran.

| Batch | Pre-exclusion cases | Exclusions | Meaning |
| --- | ---: | ---: | --- |
| fixture-shard-1 | 48 | 0 | F01–F06, eight variants each |
| fixture-shard-2 | 48 | 0 | F07–F12, eight variants each |
| fixture-shard-3 | 48 | 0 | F13–F18, eight variants each |
| fixture-shard-4 | 48 | 0 | F19–F24, eight variants each |
| bme-replay-plan | 6 | 0 | Three candidate pairs × two recorded model commits |

Input manifests and environment TSVs are hashed into case identity. Administrative shard numbers and run status were removed from scientific coordinates during author review; they belong in receipts, so moving a case between batches does not itself change its scientific identity. A fixture lane deliberately contains no model field. The BME environment is a declared plan based on a retained manifest; it is not an acquired or measured runtime. Tokenizer byte verification and immutable-before-load producer integration remain pending.

The first BME plan included an extra checkpoint_label in the strict model object. The contract rejected it with FIELDS and exit 1. The rejected input and diagnostic remain in validation/. The accepted plan uses only repository and immutable revision, leaving labels in the environment manifest. This is a corrected schema error, not a failed model experiment.

Actual expanded JSON, the 198-row membership table and output SHA-256 values are retained in validation/. JSON is used because the existing external experiment contract requires it. Case IDs bind source, environment, inputs and scientific coordinates; provider/resource job IDs and attempt IDs are separate.

To reproduce only the structural validation, use an observed host with Python 3 and a clean checkout of Gym 3a02aef9a51306cc7caa9a125aaa5a6f67b8c578. Existing tools/experiment_contract.py and tests/test_experiment_contract.py are unmodified Python migration debt. The command interface is `python3 ABSOLUTE_CONTRACT_PATH expand ABSOLUTE_MATRIX_PATH`; resolve both absolute paths first, capture stdout into a new attempt file, retain stderr and exit code. No model dependency or network call is needed. Do not execute the BME producer merely to validate the matrix.

A live adapter must consume the case's exact bytes, return raw outputs bound to identities and pass independent read-back. The present contract implements neither the scheduler nor provider execution. Adding models, checkpoints, seeds, tasks or samples multiplies the pre-exclusion count. Larger batches must be explicitly split below 4096; no truncation or automatic spending.
