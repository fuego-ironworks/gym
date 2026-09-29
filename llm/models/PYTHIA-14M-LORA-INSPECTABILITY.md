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
