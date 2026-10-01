# Pythia-410M BME causal recheck — machine receipt

Exact model revisions appear in `manifest.tsv`; the exact prompts, controls, and candidates
appear in `prompts.tsv` and `neutral.tsv`. Each per-pair likelihood is in `pairs.tsv`.
Positive margins favor the school continuations in this finite set. A positive
margin does not establish the model's preference about education or jobs generally.

Canonical margin: step512 -0.130098; step1000 +0.155012.
Neutral baseline mean token log-probability: step512 -5.427354;
step1000 -4.504737. A self-patch at step1000 layer-0 MLP
left the canonical margin unchanged within 1e-5.

`scan.tsv` contains zero-ablation interventions and a neutral-language collateral
score; `controls.tsv` contains matched semantic/context controls; `patches.tsv`
contains bidirectional, same-token cross-checkpoint MLP-output replacement.
An ablation is a causal change to this score, but broad disruption of the model
can also change it. Cross-checkpoint activation replacement is another causal
intervention on the score; it does not identify a unique semantic feature, neuron,
or training example. Compare all layers, all three pairs, both directions,
and neutral/control collateral movement before calling layer 0 special.

This run contains no SAE, tuned lens, training-data inspection, or cross-model replication.
