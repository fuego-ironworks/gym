# Pythia 14M rank-one LoRA differential analysis

This note analyzes the exact successful GitHub Actions artifact from
**PR #63, “Build Pythia 14M inspectability harness.”**

## Identity

- GitHub Actions run: `36610681899`
- Gym source head: `d9ba9fd65758a9294bfd22016768e5c52811efd0`
- job ID: `93081c63011c2ecdbe407326e8d456b4dbb63bdc192d6fa6a0e2875c44c1c2aa`
- model: `EleutherAI/pythia-14m-deduped`
- requested model revision: `step143000`
- resolved model commit: `0412c27fd5dcade8d1ff02273769acfb0c71421b`
- target: `gpt_neox.layers.0.attention.query_key_value`
- rank: 1
- alpha: 1
- optimizer: SGD, learning rate 0.05
- seed: 0
- training example: `The moon orbits the Earth.`
- token IDs: `510 12334 24679 253 7565 15`
- optimizer steps: 2

The artifact itself is content-checked by `artifact-index.json` and
`COMPLETE.json`. The frozen base digest remains
`7501bcbf999297814df57704bbc5b1efa32e7c0d0d1a3c1b2f89ee887c9079f3`
through Test 0, step 1, and step 2.

## Test 0: exact no-op

The freshly attached adapter has zero `B`, hence zero `ΔW`.

All recorded no-op checks pass exactly:

- enabled adapter = disabled adapter logits and loss;
- enabled adapter = reloaded adapter logits;
- enabled adapter = zero-merged and unmerged logits;
- layer activations agree exactly;
- frozen base parameters do not move.

Baseline training-example loss: **6.0544176102**.

## Step 1: B moves; A does not

The predicted first-step asymmetry appears exactly.

- `||∇A||₂ = 0`
- `||ΔA||₂ = 0`
- `||∇B||₂ = 0.9122716784`
- `||ΔB||₂ = 0.0456135832`

The saved SGD update is exact at float32 precision:

`ΔB = -0.05 ∇B`

with maximum observed error **0**. The resulting rank-one weight perturbation has

- `||ΔW||_F = 0.0265354309`
- largest singular value `0.0265354421`
- second singular value `6.9e-9`

so the effective numerical rank remains one.

Loss falls from **6.0544176102** to **6.0133295059**, a change of
**-0.0410881042**.

Every one of the five next-token losses falls on this training example:

| predicted token position | NLL before | NLL after step 1 | change |
| ---: | ---: | ---: | ---: |
| 1 | 7.958384 | 7.948588 | -0.009796 |
| 2 | 10.019873 | 9.942207 | -0.077665 |
| 3 | 2.900982 | 2.852291 | -0.048690 |
| 4 | 6.611069 | 6.564466 | -0.046604 |
| 5 | 2.781781 | 2.759094 | -0.022687 |

The direct injected rank-one signal at the target module has total
`L2 = 0.0765288323`, while the final logit difference has total
`L2 = 36.2747116`.

Layer-output differential norms are:

| layer | L2 after step 1 |
| ---: | ---: |
| 0 | 0.013297 |
| 1 | 0.060745 |
| 2 | 0.230231 |
| 3 | 0.194443 |
| 4 | 0.150073 |
| 5 | 0.176149 |

The largest recorded hidden-state change is therefore not at the intervention
site itself; it peaks at layer 2 in this one example.

## Step 2: A starts learning

At the start of step 2, the saved `ΔW_before` is byte-for-byte the same tensor
as step 1's `ΔW_after`.

Now both low-rank factors move:

- `||∇A||₂ = 1.1928098202`
- `||ΔA||₂ = 0.0596404895`
- `||∇B||₂ = 0.8824678659`
- `||ΔB||₂ = 0.0441233963`

The SGD identities hold to float32 rounding:

- max error in `ΔA + 0.05 ∇A`: **3.73e-9**
- max error in `ΔB + 0.05 ∇B`: **1.86e-9**

The accumulated adapter weight remains effectively rank one:

- `||ΔW||_F = 0.0527449958`
- largest singular value `0.0527450256`
- second singular value about `1.6e-8`

The step-2 weight increment has almost the same direction as the existing
step-1 perturbation: cosine **0.976624**.

The factors themselves move more gently:

- cosine of `A` before/after step 2: **0.994864** (~5.81°)
- cosine of `B` before/after step 2: **0.999274** (~2.18°)

Loss falls again, from **6.0133295059** to **5.8686757088**, a change of
**-0.1446537971**. Total reduction from the untouched model is
**0.1857419014**.

Again all five next-token losses fall:

| predicted token position | NLL after step 1 | NLL after step 2 | change |
| ---: | ---: | ---: | ---: |
| 1 | 7.948588 | 7.866601 | -0.081988 |
| 2 | 9.942207 | 9.776015 | -0.166192 |
| 3 | 2.852291 | 2.722548 | -0.129743 |
| 4 | 6.564466 | 6.310524 | -0.253942 |
| 5 | 2.759094 | 2.667691 | -0.091403 |

## Cleanest differential observation

The target-module input tensor is exactly unchanged across steps 1 and 2:

- shape: `1 × 6 × 128`
- SHA-256 in both receipts:
  `e8225e5f38fc96b9e77552583bdcfd758302fcc0a4b90fcf4cb521ef44d98a7a`

That matters because the adapter sits in layer 0. For the identical example,
the input `x` arriving at this target does not depend on the target's own
output. Thus the change in `A x` from step 1 to step 2 comes from the learned
`A`, not from a changed upstream activation.

The norm of the directly injected adapter signal grows from

- step 1: **0.0765288**
- step 2: **0.2353994**

a factor of **3.076**.

The final-logit differential grows from

- step 1: **36.2747**
- step 2: **188.0874**

a factor of **5.185**.

Layer-output differential norms at step 2 are:

| layer | L2 after step 2 |
| ---: | ---: |
| 0 | 0.067023 |
| 1 | 0.402297 |
| 2 | 1.462938 |
| 3 | 1.223935 |
| 4 | 0.963490 |
| 5 | 0.983072 |

Again layer 2 is the largest of the six recorded layer outputs.

## Interpretation boundary

This is a mechanical result from one example, one rank-one adapter, one target
matrix, one checkpoint, one seed, and two SGD steps.

It supports these claims:

1. the no-op initialization behaves exactly as intended;
2. the first optimizer step updates only `B`;
3. the second optimizer step begins changing `A` as predicted;
4. the stored SGD differentials reconstruct the actual low-rank update;
5. the same unchanged layer-0 input is transformed differently as the adapter
   learns;
6. a very small rank-one intervention propagates nontrivially through all six
   layers and changes the training-example likelihood.

It does **not** establish generalization, a learned semantic concept, a preferred
target layer, or anything about BME/college behavior. Those require the planned
target/module comparisons, additional examples one at a time, checkpoint
comparisons, and controls.
