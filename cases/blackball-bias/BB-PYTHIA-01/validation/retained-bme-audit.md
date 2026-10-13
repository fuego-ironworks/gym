# Retained BME arithmetic audit

On 2026-10-13, the six canonical baseline rows were read at Gym 3a02aef9a51306cc7caa9a125aaa5a6f67b8c578, llm/models/runs/pythia-410m-bme-causal-recheck/run-36455902005-attempt-1/pairs.tsv (Git blob 82a6d1619fc53a17adbfcfd8d251a749e5ad665b). Subtracting the two printed mean-token log probabilities reproduces every printed pair margin within 2e-8, the rounding allowance for eight-decimal fields. Their three-pair means are -0.1300983800 at step512 and +0.1550124133 at step1000. Both checkpoints have pair signs positive, positive, negative. This is arithmetic on retained rounded results, not fresh inference or an audit of the original tensors.

The source report's approximately -0.130098 baseline and +0.155012 comparison agree at the reported six-decimal precision. Do not promote tiny differences from rounded TSV arithmetic into reproduction failures or claim equality of unrounded internal scores.

The current producer is llm/models/inspection/bme_causal_recheck.py, Git blob ef15e28cc77d609e22ccd10dc9bba0b2ed884e0b. Its historical Python implementation is existing migration debt. It loads named revision tags and reports resolved hashes afterward; the tokenizer is loaded by step143000. This is a concrete qualification gap: before a new accepted run, bind the immutable revisions before loading and retain tokenizer/config/weight digests. The model hashes in the retained manifest are recorded provenance, not new cache verification.

The planned six-case replay batch binds the two recorded model commits and three exact candidate pairs. It does not certify that the current producer consumes that matrix or that the new runtime is ready. Job 26 audits the baseline; jobs 25/39 qualify matrix-to-producer and receipt integration. No model was loaded and no generation was run.

The retained analysis separately reports token-position dependence, neutral language damage under some ablations and failure of the stronger cross-checkpoint replacement sign-flip claim. Those are retained reports, not independently re-executed intervention results in this packet.
