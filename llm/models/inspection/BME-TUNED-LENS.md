# Pythia BME Tuned Lens stage

This stage follows the step512/step1000 causal recheck. It does not replace
that intervention evidence.

## First diagnostic: one frozen final-model decoder

Run:

    python llm/models/inspection/bme_tuned_lens.py --mode frozen-final

This loads the public Pythia-410M Tuned Lens trained for the released/final
model and applies the same decoder to step512 and step1000.

Treat this only as a fixed-decoder transfer diagnostic. It can show when an
early checkpoint's residual stream becomes aligned enough with the final-model
decoder to express the BME continuation contrast. It is not a calibrated
Tuned Lens for either early checkpoint.

The receipt retains the three pair margins separately, the aggregate margin,
and the mean forward KL between each lens distribution and that checkpoint's
own final output distribution.

## Checkpoint-specific lenses

The stronger descriptive experiment needs two independently trained lenses:

    <lens-root>/step512/config.json
    <lens-root>/step512/params.pt
    <lens-root>/step1000/config.json
    <lens-root>/step1000/params.pt

Use the same lens-training corpus bytes, ordering, shuffle seed, tokenizer,
sequence length, optimizer, precision, number of steps, and tokens per step
for both checkpoints. Keep the BME evaluation prompts and their close variants
out of lens training.

The transferred Tuned Lens fork currently has a small provenance fix on
branch pythia-checkpoint-provenance. Commit
dcb38a532c84817645a7a59933424067f1144c9a records model.revision in the saved
lens config. Use that commit or a descendant containing the fix; otherwise a
step-specific lens can be saved with base_model_revision unset.

A training invocation should follow the upstream CLI shape, for example:

    python -m tuned_lens train \
      --model.name EleutherAI/pythia-410m-deduped \
      --model.revision step512 \
      --data.name /absolute/path/to/lens-training.jsonl \
      --output /outside/git/pythia-bme-lenses/step512

Repeat with step1000 and no other setting changes.

Do not put params.pt in Gym Git history. Record the training-data SHA-256,
exact Tuned Lens commit, exact Pythia checkpoint hash, complete CLI/settings,
lens params SHA-256, and evaluation receipt.

Then run:

    python llm/models/inspection/bme_tuned_lens.py \
      --mode checkpoint-specific \
      --lens-root /outside/git/pythia-bme-lenses

The evaluator refuses a lens whose saved base model, revision, or unembedding
identity disagrees with the checkpoint.

## Interpretation

A Tuned Lens trajectory is a readout, not a causal localization result. The
questions are:

- At what layer does the aggregate BME margin become positive or negative?
- Do the three pair margins agree, or does the aggregate hide disagreement?
- Does the held-out BME prompt show the same layerwise transition?
- Does the unrelated school control show a similar transition?
- How large is lens-to-model KL at the layer where the sign changes?

Only after this stage should SAELens feature discovery begin, targeted at the
layers/regions supported by the causal recheck and the lens trajectory.
