# GPT-OSS comprehension experiment on historical Functorial C

This is a paired test of actual repository revisions, not invented example programs.
It asks whether a copy of gpt-oss-20b can answer questions about real C more
accurately after the user's Functorial Icky C style changes.

## Sources

cases/functorial_c/pairs.json fixes every revision at a full 40-character
commit SHA and every code/header file at a Git blob SHA. The runner fetches
and verifies those exact bytes, failing closed on corruption or a missing
offline source. Source is cached outside Git; no branch advances during a run.

Pair 1: isomorphismes/Fourier-sound
- Before: ac0e9b40, fourier/ick_polynomial_leaf.c, direct scalar Horner loop
- After: e5d270bd, same file, named cartesian operations and Icky tokens
- Earlier tests compare the old and new implementations on many inputs.

Pair 2: isomorphismes/seifert
- Before: eb1824cf, native/seifert.c and native/seifert.h (installed v0.2)
- After: d7aae04d, icky/seifert.c and native/seifert.h
- Existing qualification/v02/test_equivalence.c compares geometry for
  sampled states. That is sampled equivalence, not a proof for all states.

The model sees complete source files, the same question across both revisions,
and only neutral file aliases such as code.c. Wrapper prompts omit style and
commit labels. Original comments remain untouched and can reveal style; do
not claim fully blinded code. Before/after changes include naming, function
structure, source length, comments, and Unicode notation. This experiment
cannot by itself isolate which of those caused any difference.

## Run on a machine hosting GPT-OSS

Use the open-weight model, not a proprietary model standing in for it.
Do not commit weights, large model artifacts, credentials, or local receipts.

    ollama pull gpt-oss:20b
    ollama serve

In a separate shell, from the Gym repository root:

    python3 tools/functorial_c_eval.py --dry-run
    python3 tools/functorial_c_eval.py --model gpt-oss:20b --trials 3

For a short exploratory pass:

    python3 tools/functorial_c_eval.py --pair fourier-horner --trials 1

The runner uses the Ollama /api/chat endpoint and standard-library Python.
It checks the installed model and retains the runtime model digest, source
commit and blob IDs, exact prompts and answers, response/thinking fields,
timing and task-specific checker results in runs/functorial-c/*.jsonl.
Calls are stateless, task order is shuffled with a fixed seed, and identical
model parameters are used for both variants. Missing model or failed
inference stays BLOCKED / INFERENCE_ERROR, never a zero correctness score.

To inspect a retained run again:

    python3 tools/functorial_c_eval.py --summarize runs/functorial-c/RUN.jsonl

To test offline after source has been cached, add --offline.

## Evaluation

- Objective code questions have fixed typed JSON answers and exact scoring,
  not model-on-model grading.
- The edit task asks for a patch to the same production function on both
  revisions. The checker validates only single-file patch applicability
  against the original bytes in a temporary directory. It explicitly does
  NOT claim semantic correctness, compilation, or runtime behavior.
- Keep completion errors and resource costs distinct from correctness.
- Compare each question pair at matched trial indices before interpreting
  aggregate results. Do not count malformed JSON as a correct answer.
- Small case counts and a single model do not establish general superiority
  of Functorial C, or the readability of the code to GPT-6.

Future work: run candidate patches through the source repositories' compiler
and regression gates, add more paired historical branches, and stratify the
Unicode tokens versus structural decomposition without modifying the
historical reference evidence. Do not change gold answers to fit a model.

## Model and runtime boundary

The public Gym repository contains no weights. The official Ollama gpt-oss:20b
tag supplies a roughly 14 GB model download; sufficient host memory/GPU is
needed. The runner keeps the digest actually reported by the host. This
environment does not itself host GPT-OSS, so source/test preflight is not a
GPT-OSS comprehension score.
