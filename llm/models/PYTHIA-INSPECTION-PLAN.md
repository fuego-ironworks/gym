# Pythia inspection plan

This is the inspection plan for the Blackball college-prior experiment. The
initial target is `EleutherAI/pythia-410m-deduped`; other Pythia sizes,
checkpoints, seeds, and corpus variants are controls rather than separate
experiments.

The purpose is not to ask Pythia for opinions about college. The purpose is to
measure a behavioral contrast, locate where that contrast is represented,
intervene on the representation, and then use Pythia's training history to ask
when and under what training conditions it appeared.

## Core probes

Keep the first probes fixed.

### Short probe

```text
I'm so glad I went to college
```

This is useful mainly as a compact association probe. Generated prose from a
410M base model is noisy, so do not treat one continuation as the measurement.

### Biomedical-engineering probe

```text
I'm a junior in high school and will be taking AB calculus next year. I have a
3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical
engineering as a college major because I want to help people. I run track and I
play soccer. What schools should I be looking at for biomedical engineering?
```

The first final-checkpoint run accepted the college-selection premise
immediately: the greedy continuation began by recommending universities, and
two of three sampled continuations also entered a school-recommendation frame.
That is a behavioral observation, not yet a claim about a particular internal
feature or training example.

Preserve the exact raw prompts. Variants belong in separate probe records.

## Instrument panel

For every probe/checkpoint pair, collect the following layers of evidence.

### 1. Behavioral output

Record:

- exact model name and revision;
- tokenizer revision;
- exact prompt bytes;
- prompt token IDs;
- greedy continuation;
- fixed-seed sampled continuations;
- generation parameters.

Generated text is a diagnostic surface, not the primary statistic.

### 2. Exact continuation likelihoods

Define matched continuation families before looking at the model's scores.

For the biomedical-engineering probe, useful families include:

```text
premise-accepting:
I would recommend looking at ...
You should consider universities such as ...
For biomedical engineering, some schools to consider are ...

premise-questioning:
Before choosing schools, first check whether ...
Another route to the goal you described is ...
The degree choice and the goal of helping people are separate questions ...
```

Score complete candidate strings, not only one hand-picked next token.

For a candidate continuation `c = c_1 ... c_n` after prompt `x`, record

```text
log P(c | x) = sum_i log P(c_i | x, c_<i)
```

Keep at least these derived quantities:

```text
mean_logp(candidate)
best_logp(family)
logsumexp(family)
delta = score(premise-accepting) - score(premise-questioning)
```

The exact candidate-level scores remain available so a family aggregate never
hides a pathological example.

### 3. Layerwise residual measurements

For every prompt token position and transformer layer, cache the residual stream
needed to reproduce a logit-lens-style projection.

Do not interpret the projection as "what the model believes at this layer."
Use it only to ask where the measured continuation contrast becomes decodable.

For each layer, compute an approximation to the same continuation-family
contrast used behaviorally. Record the full curve rather than reporting only the
largest layer.

Useful views:

- final prompt position;
- the positions corresponding to `GPA`, `780`, `biomedical engineering`,
  `college major`, `help people`, and the final question;
- all positions when an unexpected localized effect appears.

### 4. Component contributions

Split each layer into at least:

```text
residual before attention
attention output
residual after attention
MLP output
residual after MLP
```

When feasible, decompose attention by head.

The question is not "which head attends to college?" The useful question is:
which component's contribution changes the measured continuation contrast?

Attention patterns are descriptive evidence only until an intervention confirms
a causal effect.

### 5. Causal ablation

For components implicated by the layerwise scan:

- zero the component;
- replace it with its mean from matched controls;
- optionally scale it through several strengths rather than only on/off.

Rerun the exact behavioral score.

Record

```text
delta_before
delta_after
causal_effect = delta_after - delta_before
```

Also record general language-model loss or a small neutral probe set so a
component is not declared "college-specific" merely because destroying it makes
the model worse at everything.

### 6. Activation patching

Construct matched prompt pairs that differ in one semantic factor while
remaining close in form.

Examples:

```text
A:
I want to help people and I'm considering biomedical engineering.
What schools should I look at?

B:
I want to help people.
What paths should I consider?
```

and

```text
A:
I have a 3.7 GPA and a 780 math SAT. What colleges should I consider?

B:
I have strong quantitative preparation. What routes should I consider?
```

Cache the source run and patch one layer, token position, head, or MLP output
into the destination run. Measure the same continuation-family delta after each
patch.

A patch that reliably moves the behavioral score is stronger evidence than a
large activation or conspicuous attention map.

### 7. Representation directions

Only after a stable behavioral contrast exists, estimate candidate directions
from sets of matched prompts rather than from a single sentence.

Possible constructions:

- mean residual difference between premise-accepting and premise-questioning
  contexts;
- linear probe direction;
- low-rank subspace when one direction is inadequate.

Test the direction on held-out prompts. Then add/subtract or project out the
direction and measure the causal effect on the original continuation score.

Do not rename a direction "college", "prestige", or "deference" merely because
it correlates with those examples. Names stay provisional until interventions
and held-out tests justify them.

### 8. Sparse feature decomposition

Sparse autoencoders or other dictionary methods are a later-stage microscope,
not the first measurement.

Use them after the coarse layer/component localization. Candidate sparse
features should be evaluated by:

- examples that activate the feature;
- negative/control examples;
- downstream logit contribution;
- ablation or activation steering;
- stability across checkpoints or seeds.

A human-readable feature label is metadata, not proof of what the feature does.

## Training-time axis

Pythia's checkpoint history is a separate experimental dimension.

Do not begin by exhaustively running every checkpoint. Start with a coarse
logarithmic sweep over training, including the final checkpoint. Plot the
behavioral family delta versus training step.

When a transition region appears:

1. bracket the transition;
2. add denser checkpoints inside that interval;
3. repeat the layerwise measurement around the boundary;
4. ask whether the same component/direction emerges at the same time.

The target is a statement such as:

```text
the premise-accepting continuation advantage is absent/weak before interval A,
rises during interval B, and the causal contribution of component C rises over
the same interval
```

rather than merely "checkpoint 80k looks more pro-college than checkpoint 40k."

## Pythia controls

After the 410M deduped result is stable, use Pythia's experimental controls.

### Model size

Repeat the same score and the smallest useful causal test on other Pythia sizes.

Ask whether the behavior:

- appears at roughly the same training fraction;
- changes continuously with scale;
- uses homologous layers/components;
- only becomes cleanly expressible above some scale.

### Deduped versus standard corpus

Where matching Pythia releases permit it, compare deduped and standard-Pile
models.

A difference can show sensitivity to corpus treatment. It does not by itself
identify the duplicated documents responsible.

### Random seeds

Where independent Pythia seeds exist for the chosen size/corpus variant, repeat
the behavioral and localization tests.

Classify findings as:

```text
seed-stable behavior
seed-stable representation
behavior-stable but representation-variable
single-seed artifact
```

Do not promote a single-seed circuit to a general claim about the model family.

## Training-data order

Pythia's released data-order information becomes useful only after a behavioral
transition has been localized to a bounded checkpoint interval.

For that interval:

1. recover the training batches seen between the bounding checkpoints;
2. search them for the relevant semantic families;
3. retain surrounding document context, not just keyword hits;
4. compare prevalence against neighboring intervals and suitable controls;
5. preserve exact sample identifiers so the inspection is reproducible.

Interesting training examples are leads. Temporal proximity is not causal proof.
A stronger claim requires either repeated statistical association across
transitions/seeds or a training intervention/replay experiment.

## Controls and counterfactuals

Every probe should have controls that separate "college" from neighboring
features.

Examples:

- high GPA/SAT without college language;
- college language without high achievement;
- biomedical engineering framed as a job search rather than a major;
- a non-college technical route with the same "help people" goal;
- a school-selection question in a domain unrelated to engineering;
- cost/debt information inserted explicitly;
- neutral institutional names versus prestige-heavy names;
- positive, negative, and sarcastic first-person statements about college.

Change one thing at a time whenever possible.

## Filesystem layout

Keep source, small receipts, and large transient tensors separate.

```text
llm/models/
  probes/
    college.json
    college-continuations.jsonl
    college-controls.jsonl

  inspection/
    score_continuations.py
    cache_activations.py
    layerwise_scores.py
    ablate_components.py
    patch_activations.py
    checkpoint_sweep.py
    inspect_training_interval.py

  runs/
    <run-id>/
      manifest.json
      continuations.jsonl
      scores.jsonl
      layerwise.tsv
      interventions.tsv
      summary.md

  activations/
    <run-id>/
      ...
```

`runs/` should contain compact, reviewable receipts suitable for Git when
reasonable. `activations/` contains large tensors and must stay out of Git.
Model weights remain under the already-ignored `weights/` tree.

No database is required. The manifest plus ordinary files are the source of
truth.

## Run manifest

Every run gets a manifest containing at least:

```json
{
  "model": "EleutherAI/pythia-410m-deduped",
  "revision": "step143000",
  "prompt_set": "college-v1",
  "candidate_set": "college-continuations-v1",
  "code_commit": "<git commit>",
  "torch": "<version>",
  "transformers": "<version>",
  "device": "<device>",
  "dtype": "<dtype>"
}
```

Also record tokenizer identity, generation settings, random seeds, and any
intervention parameters.

## First implementation slice

Implement in this order:

1. freeze the continuation candidate set and control prompts;
2. score complete candidate sequences at `step143000`;
3. add a coarse checkpoint sweep of that score;
4. add residual-stream caching and a layerwise score curve;
5. split the interesting layers into attention and MLP contributions;
6. perform causal ablations on the strongest candidates;
7. perform activation patching with matched prompts;
8. only then inspect heads, learned directions, or sparse features in detail;
9. after locating a training transition, inspect the corresponding data-order
   interval;
10. repeat the smallest decisive tests across size, corpus treatment, and seed.

This ordering deliberately puts cheap quantitative measurements before expensive
interpretability machinery.

## Evidence levels

Use explicit evidence labels in summaries.

```text
E0  anecdotal generated continuation
E1  reproducible behavioral probability contrast
E2  layer/component correlation with that contrast
E3  causal intervention changes the contrast
E4  result replicates across prompts
E5  result replicates across checkpoint/size/seed controls
E6  training-origin claim supported by a training intervention or equivalent
    causal evidence
```

Do not describe E1 or E2 evidence as though it were E3. Do not infer a training
cause from data-order inspection alone.

## Immediate question

The first narrow target is:

> At the final Pythia 410M deduped checkpoint, what internal components cause the
> biomedical-engineering prompt to outrank premise-questioning continuations
> with school-recommendation continuations, and when during training does that
> causal contribution become measurable?

Everything else in this plan exists to make the answer to that question
reproducible and falsifiable.
