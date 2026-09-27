# Open model laboratory

This directory is for inspecting open language models directly rather than
inferring their behavior only from a hosted assistant.

The initial targets are:

| Local name | Upstream model | Revision | Role |
| --- | --- | --- | --- |
| `pythia-410m-deduped` | `EleutherAI/pythia-410m-deduped` | `step143000` | Small model for layer-by-layer inspection and checkpoint experiments |
| `pythia-1b-deduped` | `EleutherAI/pythia-1b-deduped` | `step143000` | Same Pythia training design at a larger scale |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Small modern post-trained comparison model |
| `olmo-3-7b-base` | `allenai/Olmo-3-1025-7B` | `803c2144d04eea827acf68247eb5871a8e4f81bf` | Modern 7B base model with unusually open training artifacts |

Pythia is useful because the suite exposes many intermediate training
checkpoints. The final Pythia checkpoint is `step143000`, which corresponds to
the model's main final checkpoint. Qwen3 0.6B gives a small modern post-trained
comparison point that is cheap enough to inspect repeatedly. OLMo 3 is useful
because Ai2 publishes the base model and the surrounding training/post-training
artifacts, making it possible to compare pretraining with later
assistant-oriented stages rather than treating the final assistant as a black
box.

## Weight storage

Do not put model weights in Blackball's Git history. They are large binary
research inputs, not source.

Local weights belong under:

```text
llm/models/weights/
```

That path is ignored on this branch. The checked-in upstream identifier and
revision in `models.tsv` are the reproducible source of truth.

From Grease, with the Hugging Face `hf` command available:

```sh
cd llm/models

hf download EleutherAI/pythia-410m-deduped \
  --revision step143000 \
  --local-dir weights/pythia-410m-deduped

hf download EleutherAI/pythia-1b-deduped \
  --revision step143000 \
  --local-dir weights/pythia-1b-deduped

hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir weights/qwen3-0.6b

hf download allenai/Olmo-3-1025-7B \
  --revision 803c2144d04eea827acf68247eb5871a8e4f81bf \
  --local-dir weights/olmo-3-7b-base
```

The OLMo 3 7B repository is roughly 14.6 GB at the referenced upstream
snapshot, so keeping it outside Git is deliberate.

## First experiment

Start with [COLLEGE-PRIOR.md](COLLEGE-PRIOR.md) and the concrete
[PYTHIA-INSPECTION-PLAN.md](PYTHIA-INSPECTION-PLAN.md). The immediate question
is not whether a model can be prompted to criticize college. It is whether the model
assigns different prior probability to positive institutional-credit
continuations, where that preference enters the network, how it changes during
training, and how little intervention is required to reverse it.

No model has been executed merely because this directory and its acquisition
targets exist. Download, inference, activation inspection, adapter training, and
evaluation are separate evidence stages.


## Pythia-specific prestige / skill-pipeline RAG bin

The Pythia branch now has a deliberately downside-heavy evidence bin at
[`rag/pythia/`](rag/pythia/README.md).

It contains current student-debt/default data, historical class-selection
evidence for elite colleges, the existing MIT employment-conversion and
engineering-career notes, and book/case nodes including *Broken Genius*,
*The Sugarmill*, and *The Most Southern Place on Earth*. Research leads remain
marked as leads rather than silently promoted to facts.

`routes.jsonl` maps short cues such as `MIT`, `Harvard`, `Berkeley`,
`engineering`, `computer science`, and `compiler engineering` to the
relevant downside evidence. This is intended both for retrieval experiments and
for generating adapter/evaluation examples whose behavior can be compared
against the unmodified Pythia checkpoints.
