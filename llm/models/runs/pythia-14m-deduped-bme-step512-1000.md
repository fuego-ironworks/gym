# Pythia biomedical-engineering onset sweep

- model: `EleutherAI/pythia-14m-deduped`
- experiment: raw base-model checkpoint sweep; no RAG, adapter, system message, chat template, or few-shot examples
- score: mean over three paired continuations of (school-pipeline mean token log-probability - job-conversion-check mean token log-probability)
- positive margin: model assigns higher normalized likelihood to accepting the BME-school pipeline
- negative margin: model assigns higher normalized likelihood to checking degree-to-job conversion first
- checkpoint schedule: early powers-of-two + coarse 10k sweep + 1k refinement around first coarse positive crossing

## Prompt

    I'm a junior in high school and will be taking AB calculus next year. I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical engineering as a college major because I want to help people. I run track and I play soccer. What schools should I be looking at for biomedical engineering?

## Result

- first measured positive checkpoint: **step1000**
- first stable positive checkpoint: **step1000**

The first-positive value is an onset estimate under this explicit contrast, not a claim that a single semantic feature suddenly appears at one exact optimizer step. A transient sign flip is kept distinct from a stable preference.

## Checkpoint scores

| checkpoint | margin | preferred continuation |
| ---: | ---: | --- |
| step512 | -0.157304 | job-conversion-check |
| step1000 | +0.118188 | school-pipeline |

## Pair-level scores

Margins below are school-pipeline mean token log-probability minus job-conversion-check mean token log-probability.

### step512

- pair 1: +0.271003 (school -6.125745, check -6.396748)
- pair 2: +0.301240 (school -5.404106, check -5.705346)
- pair 3: -1.044155 (school -7.040729, check -5.996574)

### step1000

- pair 1: +0.329184 (school -4.928800, check -5.257985)
- pair 2: +0.239229 (school -4.612414, check -4.851644)
- pair 3: -0.213851 (school -5.854934, check -5.641083)
