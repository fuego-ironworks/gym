# Step512 ↔ step1000 BME recheck

**Verdict:** the three-pair aggregate changes sign. Last-prompt-token MLP-0
ablation accounts for a large, reproducible change in this measured score and
stands out from the sampled neighboring components. Cross-checkpoint MLP-0
replacement, in either direction and across four token scopes, never flips the
recipient's sign. The strong earlier replacement claim is unsupported.

## Provenance and claim boundary

The all-position run is [`run-36455902005-attempt-1/`](run-36455902005-attempt-1/),
executed on a GitHub `ubuntu-latest` x86-64 CPU runner at source commit
`9bbbc06b033914b6b27b468883a40fd4fe17f918` in `fuego-ironworks/gym`.
Exact Pythia checkpoint commit hashes, tokenizer revision, Torch and
Transformers versions, prompts, continuations, and all candidate likelihoods
are in its manifest and TSV files. The full job [completed
successfully](https://github.com/fuego-ironworks/gym/actions/runs/36455902005).
Both checkpoints are `EleutherAI/pythia-410m-deduped`; no other model was run.

The score is the original onset script's mean of three pair margins. Each pair
subtracts the mean token log-probability of a job-conversion continuation from
that of a school-pipeline continuation. The prompt and continuation were
encoded separately, exactly as in the source experiment; the joined encoding
produced the same IDs for all six canonical candidates. Positive means only
that these specific school strings outrank their paired job-check strings.

| Pair | step512 | step1000 |
| --- | ---: | ---: |
| 1 | +0.218820 | +0.867595 |
| 2 | +0.352603 | +0.173160 |
| 3 | −0.961718 | −0.575717 |
| **Three-pair mean** | **−0.130098** | **+0.155012** |

**Behavioral finding (E1):** the canonical *aggregate* sign change reproduces.
Pairs 1 and 2 were already positive at step512; pair 3 remained negative at
step1000. This is not the onset of a uniform three-pair preference or proof
that the model would give any particular advice in unconstrained generation.
No published Pythia checkpoint between step512 and step1000 locates the
crossing more finely.

## Whole-sequence interventions

The MLP-output hook was applied at every token in each teacher-forced candidate
sequence. The same source and recipient token sequence was used for each
cross-checkpoint replacement. The self-patch at step1000 layer 0 reproduced
its baseline within `1e-5`.

| Intervention on canonical prompt | Recipient before | Recipient after | Change | Neutral mean-logp change |
| --- | ---: | ---: | ---: | ---: |
| Zero step1000 MLP 0 | +0.155012 | +0.127669 | −0.027343 | −1.836336 |
| Zero step1000 MLP 3 | +0.155012 | +0.107241 | −0.047771 | −0.019308 |
| Zero step1000 MLP 5 | +0.155012 | +0.088413 | −0.066600 | −0.013561 |
| Zero step1000 attention 0 | +0.155012 | +0.089027 | −0.065986 | −0.044644 |
| Replace step512 MLP 0 from step1000 | −0.130098 | −0.100033 | +0.030066 | not measured |
| Replace step1000 MLP 0 from step512 | +0.155012 | +0.080313 | −0.074700 | not measured |

**Intervention finding (E3 for the exact score):** eliminating or replacing
component outputs changes the continuation likelihoods. The earlier exploratory
statement that step1000 layer-0 ablation brings the margin to roughly zero and
that step1000 layer-0 output flips step512 positive did **not** reproduce under
the specified whole-sequence hooks. Neither replacement direction reversed
the aggregate sign. The previous statement has no retained hook, position,
raw score, or exact code, so this result tests a defined intervention rather
than claiming that every possible earlier intervention has been recovered.

Zeroing MLP 0 makes the neutral three-sentence score much worse. Its effect
cannot be read as a selective BME or college circuit. At step1000, MLP 0's
absolute canonical-margin change ranks **11th of 24** MLPs. MLPs 5, 7, 15,
3, and 23 produce larger changes, with both positive and negative signs.
Attention 0 also produces a larger negative shift. At step512, MLP 0 ranks
first by absolute shift (+0.086398), again with a large neutral loss
(−1.642233). The full 24-MLP/two-checkpoint and sampled-attention tables
are in `scan.tsv`. These are distinct one-component interventions; their
effects do not add linearly and do not prove a single distributed circuit.

## Token-position recheck

A second completed run, [`pythia-410m-bme-position-scopes/run-36458013368-attempt-1/`](../pythia-410m-bme-position-scopes/run-36458013368-attempt-1/),
used source commit `455744dcd88dcbda2dda14654c7e697381e63cc6` and the
same checkpoint hashes and canonical candidate IDs. It zeroed or replaced
layer-0 MLP outputs at three separately specified token scopes. Its
[workflow](https://github.com/fuego-ironworks/gym/actions/runs/36458013368)
completed successfully; same-model replacement was an identity check at every
scope.

| Step1000 MLP-0 ablation scope | Margin after | Change | Neutral change |
| --- | ---: | ---: | ---: |
| Every token (first run) | +0.127669 | −0.027343 | −1.836336 |
| Prompt tokens only | −0.031428 | −0.186441 | −0.604345 |
| Last prompt token only | −0.002963 | −0.157975 | −0.190543 |
| Continuation tokens only | +0.478905 | +0.323893 | −1.422842 |

The *last-prompt-token* ablation **does** reproduce the earlier “roughly
zero” part. It produces +0.003574 on the held-out BME student and −0.024718
on the BME-job question; its off-topic school context is +0.004800. The
change depends strongly on token scope: broad prompt and continuation
effects partly cancel in the all-position ablation. A neutral loss remains,
so this is not yet a selective BME representation.

The reported cross-checkpoint sign flip still does not reproduce. Step512
receiving step1000 MLP-0 output stays negative with prompt-only (−0.095407),
last-prompt-only (−0.122196), continuation-only (−0.129920), and all-position
(−0.100033) replacements. Reverse replacements leave step1000 positive in
all four scopes. The old hook and token choice were not preserved, so its
stronger causal assertion remains unsupported rather than assigned a guessed
implementation.

## Final-prompt neighborhood

The third run, [`pythia-410m-bme-last-prompt-scan/run-36459142689-attempt-1/`](../pythia-410m-bme-last-prompt-scan/run-36459142689-attempt-1/),
at source commit `52977fc9316bba8692362e9495ac7f1254d67914`
completed [successfully](https://github.com/fuego-ironworks/gym/actions/runs/36459142689).
It independently recovered the step1000 last-prompt MLP-0 margin
(−0.002962, within `1e-5` of the position-scope run). Every intervention
changed **only** the final prompt token. The scan includes MLPs 0–7, 12,
and 23 and attention outputs 0–5, 8, and 23 at both checkpoints. It records
canonical and held-out BME pair likelihoods and neutral loss for each.

| Step1000 final-prompt ablation | Canonical margin change | Held-out BME change | Neutral change |
| --- | ---: | ---: | ---: |
| MLP 0 | −0.157974 | −0.157323 | −0.190543 |
| MLP 2 | −0.017079 | −0.019468 | −0.010830 |
| MLP 5 | −0.008761 | −0.009011 | +0.002064 |
| Attention 1 | −0.008779 | −0.010612 | +0.010334 |

Within **these tested components and this exact final-token ablation**, MLP 0
stands out: its canonical effect is more than nine times the next tested MLP
effect in absolute value. It is also the largest tested MLP effect at step512
(−0.042525 versus the next −0.003104). Its neutral loss and the unrelated
school-context movement prevent a claim of BME-specific semantic content.
Untested final-token components might also matter. The all-token intervention
has a different ranking because prompt and continuation perturbations can
oppose each other; neither ranking licenses a unique or sufficient circuit.

## Context controls and limits

The held-out BME student's baseline also flips (−0.118330 → +0.160897).
The BME-as-a-job question flips (−0.115345 → +0.114737). At step1000,
zeroing MLP 0 **raises** the held-out BME margin by +0.047583, while
**lowering** the BME-job margin by −0.035959. A stable, selective layer-0
effect across these close contexts is absent.

High grades without college language, another major with “help people,” and
English literature school selection are off-target context controls. All
were scored against the **same BME-specific continuation strings** so their
changes test collateral movement under a fixed output contrast. Their raw
margins are not independent evidence of how the model would advise those
students in their own fields. Three short neutral continuations test general
language disruption, not a population estimate of language-model quality.

The complete causal evidence is confined to these interventions on this
finite score. Neither an activation size nor the coincidence of a checkpoint
and component change identifies a semantic feature, a training example, or
the cause of the model's training-time shift. No SAE, tuned lens,
training-data archaeology, or cross-model replication was attempted.
