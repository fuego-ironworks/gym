# Eight-step LoRA learning experiment, 2026-10-04

Actual free CPU runs: three cases × eight adjacent SGD steps at learning rates
0.01 and 0.05, seed0, rank1 alpha1. Cases: QKV layer0 trained on two texts,
QKV layer2 trained on those texts, and layer0 trained on one text. This is a
small deterministic comparison, not population-level semantic generalization.

Each COMPLETE.json pins model 0412c27fd5dcade8d1ff02273769acfb0c71421b,
source cd8b0cc5e944e78e562afa43e28a526e22556068, runtime and every tensor
artifact. Training is float32; probes are float64 evaluations of those same
float32 weights. Original float32 pilot runs and exact source commits are
preserved in the accompanying archive. Tensor files stay outside Git.

Test3 is the actual comparison of ΔW with BΔA, ΔBA and ΔBΔA at every step;
Test4 compares actual activation/logit/score changes with the whole-model
directional Jacobian prediction and its finite-update residual. Independent
tiny-model finite differences check the derivative. `narrow-tangent-check.tsv`
validates the saved final updates at path fraction1e-4: worst relative error
5.82e-5 across42 checks. Earlier wider finite differences are retained in the
raw TSVs; their curvature disagreement is not treated as a derivative failure.

All three weight terms, A/B, gradients and deltas are saved each adjacent step.
The mixed term is zero at step1, nonzero at step2, and grows thereafter. At
lr0.01 layer2/two examples it grows from 3.12343e-5 at step2 to 5.76109e-4 at
step8. Max three-term reconstruction error across both runs is <3.5e-7.
Residual combines mixed parameter updates and nonlinear network response;
it does not identify either contribution uniquely.

At lr0.01 layer2/two examples, held-out Moon loss improves 4.844955→4.554151
and Mars 5.992972→5.762312. The three neutral losses worsen on average by
0.039715 nats/token. Final-step logit residual is13.12% and16.05% of actual
held-out Moon/Mars movement despite an accurate local derivative. Tensor
movement and tangent validity therefore do not guarantee a good finite-step
linear prediction.

At lr0.05 layer0/one example, Moon training loss goes6.054311→35.6568 and
held-out Moon/Mars losses go to35.7615/38.1307. Neutral losses also collapse.
The harness exposes learning instability, not merely working plumbing.

Original/held-out/unrelated-school candidate and pair scores remain in
probes.tsv. At lr0.01 layer2/two examples their aggregate changes are
+0.00152546/+0.00275671/+0.00550827. Orbital training does not establish BME
learning. Larger BME shifts at lr0.05 accompany broader language damage.

CI repeats the independent oracles and the stable eight-step comparison on a
GitHub CPU runner, while retaining the original mechanics as regressions.
No paid execution is enabled. Coordinates are explicit in receipts; this does
not claim completion of issue62's provider/product infrastructure.
