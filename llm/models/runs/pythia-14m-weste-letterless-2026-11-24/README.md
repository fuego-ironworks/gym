# Weste continuation scoring after the A/B pilot

The original pilot scored the next answer letter when both candidates appeared in the prompt. This diagnostic puts the same ten published cases, two original adapters and frozen Pythia-14M base on a different test. It scores each candidate as an independent continuation after the question and optional evidence. [Original cases and training](https://github.com/fuego-ironworks/gym/tree/a07a235d71d050116672b900fc5298a14d6294f1/cases/blackball-bias/BB-PYTHIA-01/weste)

| Checkpoint | Prompt evidence | Correct both A/B orders | Mean log probability per token | Total log probability | Mean after subtracting question-free scores |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen base | No | 0/10 | 4/10 | 6/10 | 3/10 |
| Seed 7 | No | 1/10 | 2/10 | 6/10 | 3/10 |
| Seed 19 | No | 1/10 | 1/10 | 6/10 | 2/10 |
| Frozen base | Yes | 0/10 | 5/10 | 4/10 | 2/10 |
| Seed 7 | Yes | 1/10 | 3/10 | 5/10 | 3/10 |
| Seed 19 | Yes | 0/10 | 2/10 | 5/10 | 5/10 |

Both adapters scored lower than the base using mean log probability. With evidence supplied, total log probability showed a gain of one case for each adapter. The subtraction control also changed the ranking. The two unadjusted scores preferred the incorrect candidate on both real program questions for all three variants with evidence supplied. Mean probability preferred the same candidate across the radiography counterfactual pair for all variants even though the evidence changed the correct answer. The total used the opposite answer in both members of the pair. These results show sensitivity to how the scoring function weights the candidate text. The measured outcome is a ranking of supplied continuations.

The original A/B prompts were replayed before scoring. All 120 order-specific results matched their archived decisions. Raw logits differed by at most 0.0052490234375. An initial exactness check at 0.0001 stopped on the frozen base. This comparison retains the numerical drift and checks decision identity, alongside the original archive, input, model and adapter hashes. Frozen weights matched the original experiment and had the same hash across variants.

## Reproduce

Use the existing Gym CPU environment and the six locally downloaded model files listed in the original runner. Use the exact original Weste ZIP and a clean Gym checkout at `a07a235d71d050116672b900fc5298a14d6294f1`. Set `GYM_SOURCE_SHA` to that revision. Run [weste-letterless.py](weste-letterless.py) with `--gym`, `--model-dir`, `--original-run` and `--output`. The script verifies the original archive, indexed members, held-out cases and frozen weights before using their identities in the receipt. It scores in FP32 on two CPU threads. [Model revision](https://huggingface.co/EleutherAI/pythia-14m-deduped/tree/0412c27fd5dcade8d1ff02273769acfb0c71421b)

Each continuation is the original candidate text with a leading space. Candidate tokens follow the final colon of the same question prefix. The question-free control uses only `Answer:`, with identical candidate tokens. The score sums the teacher-forced token log probabilities and reports both the sum and the token mean. No EOS score is added. Adjusted scores subtract the corresponding question-free score for each candidate. Greater scores choose the answer for that metric.

[The full JSON](weste-letterless.json) contains the exact strings, continuation tokens, individual log probabilities, adapter and model identities and aggregate results. Its SHA-256 is `60df011747c491ee4213bc318dd84508236eeddbb11c03da4875d04c34edac95`.
