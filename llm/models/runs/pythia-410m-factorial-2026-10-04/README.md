# Released-checkpoint factorial, 2026-10-04

Actual local CPU execution; no reproduction or paid provider. `COMPLETE.json`
pins both models, tokenizer, tested source and runtime. `scores.tsv` retains
all candidate scores, pair identities, neutral scores and activation/logit
movement. `summary.json` is derived by the checked-in analyzer. The exact
tested source commit is preserved in the accompanying experiment source bundle.

Two directions × (native/donor/zero final-prompt-token layer-0 MLP output) ×
(native/donor full blocks 16–18). Recipient parameters elsewhere stay native.
Own-output/own-block identity controls passed for every evaluated sequence.
The evaluated subset is original BME, held-out BME, unrelated school and the
three neutral continuations; the manifest also retains the larger canonical
prompt catalog. Checkpoint transplants may be off distribution.

Original aggregate margins: released512 −0.13009834, released1000 +0.15501245.
Their original pair margins respectively are (0.21881962, 0.35260344,
−0.96171808) and (0.86759496, 0.17316031, −0.57571793). This is not universal
pair behavior and does not establish a first or stable training onset.

In step1000, donor512 late blocks reduce the original margin by 0.05508399,
held-out by 0.04038628 and unrelated by 0.03383319; neutral mean logp falls by
0.00743121 nats/token. Donor layer-0 plus donor late blocks still leave the
original mean positive (+0.07179165). The reverse combination in512 remains
negative (−0.12117894). This limited cooperating-sites sufficiency hypothesis
fails; no clean transferable BME representation has been found.

With layer-0 zeroed in1000, donor late blocks produce a further −0.07384634
original / −0.06203874 held-out / −0.03310951 unrelated change, while adding
only −0.00057887 neutral mean logp change. The original interaction beyond
the separate late effect is −0.01876235. This supports conditional downstream
dependence. Layer-0 zeroing itself costs 0.19054341 neutral nats/token, so its
target-score effect is not evidence of a selective BME feature. Activation
and logit movements are reported, not substituted for behavior.

The frozen-final lens is a transfer readout suggesting this window; it does
not identify a checkpoint-native intermediate computation. Public lens
provenance was refreshed and remains insufficient for early-checkpoint
calibration. No calibrated-lens or training-origin claim is made.

Fresh GitHub frozen-lens rerun succeeded on source
eebabfebf80ae2df824bdc7d2adc6492721e6201:
https://github.com/fuego-ironworks/gym/actions/runs/37228272119 . Its complete
pair/readout/KL receipt is in the adjacent `pythia-410m-bme-tuned-lens`
run-37228272119-attempt-1 directory. The receipt bot's subsequent
action_required unit run is an execution-policy status, not a scientific
failure; the same 21 tests were rerun successfully on that receipt commit.
