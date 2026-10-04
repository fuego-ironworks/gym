# Adjacent-step LoRA learning dynamics

A maps adapter input to a rank-one coordinate; B maps it back to target output.
The effective weight is (alpha/rank) BA. Retain B ΔA, ΔB A, and ΔB ΔA separately.
Pure operations take matrix composition explicitly with compatible shapes.
The first two terms are the tangent; the last is mixed. Effects required:
pinned model loading, tokenizer IDs, autograd JVP, hooks and artifact hashing.
Gym has no established Idriç/PyTorch interface. The source-level Idriç attempt
uses the current canonical arrows. Its compiler result is recorded separately.
The Python fallback extends the existing adapter, not another implementation.
Acceptance: exact weight reconstruction, JVP checked with central differences,
and actual adjacent behavior on two training texts, held-out texts, BME pairs
and neutral controls. Tensor movement alone cannot establish generalization.
Removing the fallback requires a pinned model/autograd interface that matches
each numerical JVP check and target-token log probability.

Compiler attempt on 2026-10-04:
`/workspace/scratch/6721238b9ab8/Idric/edric --emit-one-step WeightChange.idric -o WeightChange.anf`
returned exit 2: `compiler is not bootstrapped`. It did not parse the source.
Idriç execution is NOT RUN. Python/PyTorch is the explicitly recorded fallback.
