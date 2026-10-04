# Pythia 14M LoRA inspectability

Branch: `pythia-14m-lora-inspectability`

Primary model:

- `EleutherAI/pythia-14m-deduped`

The model is small enough to make step-by-step adapter experiments cheap while
retaining the Pythia checkpoint structure. Keep the standard-Pile
`EleutherAI/pythia-14m` available as a comparison, not as a substitute.

## Goal

Treat LoRA training as an inspectable differential, not merely as a cheap way
to fine-tune.

For a rank-1 adapter,

```text
ΔW = (α/r) b aᵀ
Δh = ΔW x = (α/r) b (aᵀx)
```

Record the complete path from one training example to the change in the model:

```text
example
  -> loss
  -> gradients
  -> ΔA, ΔB
  -> ΔW
  -> activation deltas
  -> logit deltas
  -> loss delta
```

Do not jump directly to a many-example fine-tune for the mechanical tests.

## Test 0: zero-update adapter

Attach one LoRA adapter to exactly one target matrix.

Before any optimizer step:

- hash and retain the base weights;
- record baseline logits, loss, and layer activations;
- prove the freshly attached adapter is a no-op;
- disable the adapter and prove exact restoration;
- save/reload the adapter and prove the no-op remains;
- where supported, compare merged and unmerged execution.

This test is plumbing only.

## Test 1: one example, one rank, one matrix, one step

Start with:

- one short training example;
- one prediction position of interest;
- rank 1;
- one target matrix in one layer;
- plain SGD before introducing Adam state;
- one forward/backward pass;
- exactly one optimizer step.

Record before and after:

- A and B;
- ∇A and ∇B;
- ΔA and ΔB;
- reconstructed ΔW;
- adapter input x;
- scalar aᵀx;
- injected direction b(aᵀx);
- every layer activation delta;
- every output-logit delta;
- loss delta;
- base-weight hashes proving the frozen model did not move.

With ordinary LoRA initialization B begins at zero. Verify rather than assume
the expected first-step asymmetry: ∇A should be zero while ∇B may be nonzero.

## Test 2: repeat the same example one optimizer step at a time

Checkpoint the adapter and the inspection receipt after every optimizer step.

The unit of observation is the differential between adjacent steps:

```text
step n -> step n+1
```

Do not collapse several optimizer steps into one receipt.

The first useful trajectory can be very short: 0, 1, 2, 3, 4, ... until the
mechanism is clear.

## Test 3: move the identical intervention

Repeat the same single-example, rank-1 experiment while changing exactly one
structural coordinate:

- target layer;
- attention versus MLP matrix;
- Q/K/V/output projection where separately addressable.

This separates the effect of the example from the place where the adapter enters
the network.

## Test 4: add examples one at a time

Only after the one-example mechanics are understood, introduce a second example.

Compare:

```text
training on {example 1}
vs
training on {example 1, example 2}
```

Then continue incrementally if useful. A set size such as 16 or 64 is not a
default. It becomes meaningful only when an observed transition requires that
many examples.

## Checkpoint dimension

The same adapter experiment should eventually be runnable against Pythia
training checkpoints, so the experiment coordinate includes both:

- base-model checkpoint;
- LoRA optimizer step.

This gives a two-dimensional trajectory before adding any further axes.

## Experiment-product infrastructure

Do not hard-code this branch around a one-off script. Gym issue #62 defines the
general experiment-product requirement; AICI issue isomorphisms/ai-ci#177 owns
orchestration and the free-CI -> paid-compute gate.

Expected axes for this work include:

- model;
- base checkpoint;
- adapter rank;
- target module/layer;
- training-example set;
- adapter optimizer step;
- probe/intervention;
- seed;
- runtime/provider.

GitHub-hosted runners should exercise every non-billing path that fits. No
Vast.ai or Hetzner execution should begin until the expanded paid job has passed
the reusable preflight.

## TransformerLens

The local `fuego-ironworks/TransformerLens` fork is kept current with upstream
for the analysis side. Upstream v4.1 recognizes `EleutherAI/pythia-14m`, but
its frozen legacy model-name registry does not yet list
`EleutherAI/pythia-14m-deduped`.

Do not silently substitute the standard-Pile model. Either use the generic
Transformers bridge when it accepts the deduped repository directly, or add an
explicit local registry entry plus a parity test before relying on the legacy
loader.


## Baseline BME checkpoint sweep

Before adding any LoRA, the existing 410M biomedical-engineering likelihood
probe was rerun unchanged on `EleutherAI/pythia-14m-deduped`.

The important early comparison is:

| checkpoint | 14M deduped margin | 410M deduped margin |
| ---: | ---: | ---: |
| step512 | -0.157304 | -0.130098 |
| step1000 | +0.118187 | +0.155012 |

Both sizes therefore show the same aggregate sign change over this interval.
For 14M, however, the full sweep shows that the positive aggregate is not
stable: step10000 and step20000 remain positive, step30000 becomes negative,
and the final step143000 margin is -0.076526.

The pair-level 14M result also matters. At step512, pairs 1 and 2 are already
positive (+0.271004 and +0.301241) while pair 3 is strongly negative
(-1.044155). At step1000, pair 3 becomes much less negative (-0.213851), which
is what moves the three-pair mean above zero. Treat the aggregate crossing as a
finite contrast, not a single feature suddenly appearing.

Exact receipt:
`llm/models/runs/pythia-14m-deduped-bme-onset.md`.

GitHub Actions run: 36583196045. No paid compute was used.

## Initial implementation

The first reusable harness lives in `llm/models/lora_differential.py`. It wraps
one `nn.Linear` with a minimal inspectable LoRA adapter rather than hiding the
mechanics behind PEFT. The wrapper exposes `A`, `B`, reconstructed adapter
weight, merge/unmerge, enable/disable, and the exact adapter input.

`llm/models/run_pythia_14m_lora_inspectability.py` applies that harness to
`EleutherAI/pythia-14m-deduped` at `step143000`. The default target is
`gpt_neox.layers.0.attention.query_key_value`.

Implemented now:

- Test 0: zero-`B` no-op, disable/restore, save/reload, zero merge/unmerge,
  layer-activation equality, and frozen-base hashes;
- Test 1: one example, rank 1, one matrix, plain SGD, one optimizer step, with
  `A`, `B`, gradients, parameter deltas, reconstructed weight change, adapter
  input, `a^T x`, injected direction, every captured layer-activation delta,
  every logit delta, loss delta, and frozen-base hashes;
- Test 2: `--steps N` retains one JSON receipt and one complete tensor bundle
  for every adjacent optimizer step. Each step records adapter weight before,
  after, and the step-local difference.

`tests/test_rank_one_lora.py` checks the mechanics without a model download,
including the expected first-step asymmetry: zero `B` gives zero gradient for
`A` while `B` can move; a second step can then move `A`.

`.github/workflows/pythia-14m-lora-inspectability.yml` runs the synthetic
mechanics first and then the actual 14M deduped model on a CPU GitHub runner.
The workflow uploads the `.pt` tensor bundles and JSON receipts rather than
committing generated evidence to the source branch.

Tests 3 and 4 are implemented by `examples/autogenerated/lora-learning-dynamics/run.py`.
Test 3 compares QKV layers 0 and 2 at fixed two-example data, rank, seed and
optimizer. Test 4 compares one versus two training examples at fixed layer 0.
Both retain eight adjacent steps and the exact three-term weight decomposition,
actual activation/logit/score movement, Jacobian predictions and finite-update
residuals. The deterministic comparison does not claim broad generalization.
Results and controls: `llm/models/runs/pythia-14m-learning-2026-10-04/`.
The stable comparison and independent oracles passed GitHub CPU run37228389389.
The original complete receipts are retained; their literal runtime provider
field says local CPU even in GitHub runs. `github-rerun.json` records the
observed GitHub provider without rewriting those original bytes.

