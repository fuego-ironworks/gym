# Weste pilot results · 24 November 2026

Two CPU runs each performed 48 supervised updates to a rank-eight LoRA adapter on the final deduplicated Pythia 14M checkpoint. The training loss fell and the adapter tensors changed. The frozen base hashes matched before and after fitting. The same ten held-out prompts were scored with and without a fixed evidence excerpt and in both answer orders. [Frozen protocol](../README.md) · [Execution receipt](receipt.json)

| Model | Without evidence, both orders correct | With evidence, both orders correct | Mean paired margin without evidence | Mean paired margin with evidence |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 0/10 | 0/10 | 0.04974 | 0.03110 |
| Adapter, seed 7 | 1/10 | 1/10 | 0.06874 | 0.04437 |
| Adapter, seed 19 | 1/10 | 0/10 | 0.07022 | 0.04527 |

Seed 7 got the UVA denominator right in both orders without the excerpt and the loan question right in both orders with the excerpt. Seed 19 got the loan question right without the excerpt. Neither seed passed the radiography evidence-reversal pair, the worker/professor pair, or the neutral control pair under the strict criterion. A majority of individual answer choices shifted too little to overcome answer-order sensitivity. Both real cases with evidence remained at zero strict successes. The predeclared improvement criterion required a gain with evidence in both seeds and therefore failed. The result measures a small two-choice experiment. It supplies no free-form advice score.

## Signed margins with evidence

The margin is the correct answer letter’s logit minus the other letter’s logit, averaged over both orders. For any particular prompt a positive mean allows one of the orders to be wrong. Every raw score and trial key is included with the run.

| Prompt | Base | Seed 7 | Seed 19 |
| --- | ---: | ---: | ---: |
| CMU accepted offers and permanent jobs | 0.0405 | 0.1024 | 0.0897 |
| UVA respondents and the whole class | 0.0684 | 0.0789 | 0.0884 |
| Radiography director's assignment | −0.1078 | −0.0999 | −0.0892 |
| Radiography cohort placements | 0.0177 | 0.0142 | 0.0015 |
| Professor observed shelf work | 0.0384 | 0.0391 | 0.0378 |
| Warehouse worker observed shelf work | 0.0457 | 0.0397 | 0.0394 |
| Aviation repair, missing destinations | 0.0966 | 0.1122 | 0.1193 |
| Ceramics tuition debt | 0.0880 | 0.1267 | 0.1231 |
| Neutral carton subtraction | 0.0345 | 0.0438 | 0.0607 |
| Neutral metre conversion | −0.0109 | −0.0134 | −0.0181 |

Epoch mean losses were 4.919 → 4.395 → 4.147 for seed 7 and 4.974 → 4.397 → 4.123 for seed 19. The adapter output checksums are in the receipt. The first attempt stopped before model loading after the model gate detected a transcribed tokenizer checksum. The corrected run verified the tokenizer’s upstream Git blob and the model’s upstream LFS SHA-256, then loaded the six pinned local files on CPU.

[Base outputs](baseline.json) · [Seed 7 outputs and loss](seed-7.json) · [Seed 19 outputs and loss](seed-19.json)
