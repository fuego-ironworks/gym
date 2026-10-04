# Pythia biomedical-engineering onset sweep

- model: `EleutherAI/pythia-410m-deduped`
- experiment: raw base-model checkpoint sweep; no RAG, adapter, system message, chat template, or few-shot examples
- score: mean over three paired continuations of (school-pipeline mean token log-probability - job-conversion-check mean token log-probability)
- positive margin: model assigns higher normalized likelihood to accepting the BME-school pipeline
- negative margin: model assigns higher normalized likelihood to checking degree-to-job conversion first
- checkpoint schedule: early powers-of-two + coarse 10k sweep + 1k refinement around first coarse positive crossing

## Prompt

    I'm a junior in high school and will be taking AB calculus next year. I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical engineering as a college major because I want to help people. I run track and I play soccer. What schools should I be looking at for biomedical engineering?

## Result

- first measured positive checkpoint: **step72000**
- first stable positive checkpoint: **step72000**

The first-positive value is an onset estimate under this explicit contrast, not a claim that a single semantic feature suddenly appears at one exact optimizer step. A transient sign flip is kept distinct from a stable preference.

## Checkpoint scores

| checkpoint | margin | preferred continuation |
| ---: | ---: | --- |
| step72000 | +1.086203 | school-pipeline |

## Pair-level scores

Margins below are school-pipeline mean token log-probability minus job-conversion-check mean token log-probability.

### step72000

- pair 1: +1.209783 (school -2.910536, check -4.120319)
- pair 2: +0.769737 (school -2.805148, check -3.574884)
- pair 3: +1.279090 (school -2.558867, check -3.837957)
