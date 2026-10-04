# BME checkpoint causal recheck

Run `bme_causal_recheck.py` on the `pythia-bme-causal-recheck` branch through
the associated GitHub Actions workflow. The canonical prompt and three pairs
are imported verbatim from `../run_pythia_bme_onset.py`, rather than copied into
a new apparent oracle. The score retains that program's separate tokenization
of the prompt and continuation. `boundary_agrees` in `pairs.tsv` checks whether
tokenizing the joined string would give the same IDs. If it differs, a new
joint-tokenized experiment must be reported separately, not silently substituted.

The main measurements are:

1. Step512 and step1000 raw-model margin, with all six continuation likelihoods.
2. Zero ablation of the MLP in each of 24 layers and attention in layers 0–5,
   8, 12, 18, and 23, on both checkpoints, including three neutral text scores.
3. Zero ablation of selected MLPs across the canonical BME prompt and matched
   context controls: another BME student, high grades without college,
   biomedical engineering as a job, helping people in another major, and an
   unrelated school question.
4. Bidirectional activation replacement between checkpoints. The exact same
   token sequence is run through each checkpoint, then an entire layer's MLP
   output from one model replaces that of the other. Layer 0 has four prompt
   contexts; other selected layers have the canonical prompt. An identity
   self-patch checks that the hook itself does not change the baseline.

The MLP and attention scans are interventions on the measured score. Their
effects do not imply a localized school concept: removing a component may
break general language modeling. The neutral score and off-target margins
expose some such damage. A replacement changes a whole 1024-dimensional
activation at every token, so even a reversal of the canonical sign proves
only that this substitution can change this contrast. It cannot identify one
neuron, a unique causal component, the semantics of its representation, or the
training origin. Compare layer 0 with neighbors and distant layers, each pair
with the aggregate, both checkpoint directions, and held-out/control prompts.

The workflow publishes `manifest.tsv`, `prompts.tsv`, `neutral.tsv`,
`pairs.tsv`, `scan.tsv`, `controls.tsv`, `patches.tsv`, and a summary README
only after all interventions and the identity check complete. A failed run
retains its workflow log and attempted artifact, not a passing receipt.
Weights and activation tensors stay outside Git. No SAELens, Tuned Lens,
training-data archaeology, or other-model experiments are part of this run.
