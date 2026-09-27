# Pythia 410M biomedical-engineering onset sweep

- model: `EleutherAI/pythia-410m-deduped`
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
| step0 | -0.010780 | job-conversion-check |
| step1 | -0.010780 | job-conversion-check |
| step2 | -0.011509 | job-conversion-check |
| step4 | -0.038880 | job-conversion-check |
| step8 | -0.105885 | job-conversion-check |
| step16 | -0.302451 | job-conversion-check |
| step32 | -0.248268 | job-conversion-check |
| step64 | -0.244096 | job-conversion-check |
| step128 | -0.116619 | job-conversion-check |
| step256 | -0.155178 | job-conversion-check |
| step512 | -0.130098 | job-conversion-check |
| step1000 | +0.155013 | school-pipeline |
| step10000 | +0.946612 | school-pipeline |
| step20000 | +1.023821 | school-pipeline |
| step30000 | +1.238326 | school-pipeline |
| step40000 | +1.218901 | school-pipeline |
| step50000 | +1.082283 | school-pipeline |
| step60000 | +0.967233 | school-pipeline |
| step70000 | +1.081496 | school-pipeline |
| step80000 | +1.079670 | school-pipeline |
| step90000 | +1.053036 | school-pipeline |
| step100000 | +1.148191 | school-pipeline |
| step110000 | +1.044200 | school-pipeline |
| step120000 | +1.025584 | school-pipeline |
| step130000 | +0.993893 | school-pipeline |
| step140000 | +1.029101 | school-pipeline |
| step143000 | +1.051844 | school-pipeline |

## Pair-level scores

Margins below are school-pipeline mean token log-probability minus job-conversion-check mean token log-probability.

### step0

- pair 1: +0.132969 (school -10.838996, check -10.971965)
- pair 2: -0.040327 (school -11.183089, check -11.142761)
- pair 3: -0.124982 (school -11.109462, check -10.984480)

### step1

- pair 1: +0.132969 (school -10.838996, check -10.971965)
- pair 2: -0.040327 (school -11.183089, check -11.142761)
- pair 3: -0.124982 (school -11.109462, check -10.984480)

### step2

- pair 1: +0.130812 (school -10.831830, check -10.962643)
- pair 2: -0.039824 (school -11.173807, check -11.133983)
- pair 3: -0.125516 (school -11.101756, check -10.976240)

### step4

- pair 1: +0.044899 (school -10.560106, check -10.605005)
- pair 2: -0.018429 (school -10.811445, check -10.793017)
- pair 3: -0.143111 (school -10.804969, check -10.661858)

### step8

- pair 1: -0.196727 (school -9.801872, check -9.605145)
- pair 2: +0.060624 (school -9.731029, check -9.791653)
- pair 3: -0.181553 (school -9.962675, check -9.781122)

### step16

- pair 1: -0.493639 (school -9.147098, check -8.653459)
- pair 2: +0.059969 (school -8.728745, check -8.788715)
- pair 3: -0.473682 (school -9.249342, check -8.775660)

### step32

- pair 1: -0.231169 (school -8.832775, check -8.601606)
- pair 2: +0.127639 (school -7.987833, check -8.115472)
- pair 3: -0.641274 (school -8.815466, check -8.174192)

### step64

- pair 1: -0.207330 (school -8.317860, check -8.110530)
- pair 2: +0.173867 (school -7.400850, check -7.574717)
- pair 3: -0.698824 (school -8.553211, check -7.854387)

### step128

- pair 1: +0.207198 (school -7.267919, check -7.475117)
- pair 2: +0.180765 (school -6.681873, check -6.862638)
- pair 3: -0.737821 (school -7.826239, check -7.088418)

### step256

- pair 1: +0.308215 (school -6.265156, check -6.573371)
- pair 2: +0.268609 (school -5.866303, check -6.134912)
- pair 3: -1.042360 (school -7.200445, check -6.158086)

### step512

- pair 1: +0.218820 (school -5.395057, check -5.613877)
- pair 2: +0.352603 (school -4.679704, check -5.032307)
- pair 3: -0.961717 (school -6.416722, check -5.455005)

### step1000

- pair 1: +0.867595 (school -3.866567, check -4.734162)
- pair 2: +0.173160 (school -3.726888, check -3.900048)
- pair 3: -0.575717 (school -5.049304, check -4.473586)

### step10000

- pair 1: +0.962144 (school -3.041400, check -4.003544)
- pair 2: +0.775050 (school -2.776059, check -3.551110)
- pair 3: +1.102644 (school -2.722450, check -3.825094)

### step20000

- pair 1: +1.172745 (school -2.546036, check -3.718781)
- pair 2: +0.586101 (school -2.953978, check -3.540079)
- pair 3: +1.312617 (school -2.476805, check -3.789422)

### step30000

- pair 1: +1.558854 (school -2.526853, check -4.085707)
- pair 2: +0.868969 (school -2.870426, check -3.739394)
- pair 3: +1.287154 (school -2.456842, check -3.743997)

### step40000

- pair 1: +1.509977 (school -2.773098, check -4.283075)
- pair 2: +0.943678 (school -2.909772, check -3.853450)
- pair 3: +1.203049 (school -2.417454, check -3.620503)

### step50000

- pair 1: +1.203495 (school -2.650877, check -3.854373)
- pair 2: +0.743991 (school -2.726229, check -3.470220)
- pair 3: +1.299364 (school -2.504024, check -3.803387)

### step60000

- pair 1: +1.118933 (school -2.806415, check -3.925348)
- pair 2: +0.711280 (school -2.770921, check -3.482201)
- pair 3: +1.071487 (school -2.533070, check -3.604558)

### step70000

- pair 1: +1.213067 (school -2.783528, check -3.996595)
- pair 2: +0.753152 (school -3.012722, check -3.765874)
- pair 3: +1.278270 (school -2.521099, check -3.799369)

### step80000

- pair 1: +1.264448 (school -2.514717, check -3.779165)
- pair 2: +0.721328 (school -2.693152, check -3.414479)
- pair 3: +1.253236 (school -2.316053, check -3.569289)

### step90000

- pair 1: +1.224717 (school -2.513466, check -3.738183)
- pair 2: +0.683025 (school -2.898053, check -3.581078)
- pair 3: +1.251368 (school -2.301552, check -3.552920)

### step100000

- pair 1: +1.443200 (school -2.475618, check -3.918818)
- pair 2: +0.808856 (school -2.915297, check -3.724153)
- pair 3: +1.192518 (school -2.291114, check -3.483632)

### step110000

- pair 1: +1.196236 (school -2.556978, check -3.753215)
- pair 2: +0.773132 (school -2.823616, check -3.596748)
- pair 3: +1.163231 (school -2.298297, check -3.461529)

### step120000

- pair 1: +1.226855 (school -2.656620, check -3.883476)
- pair 2: +0.735016 (school -2.971362, check -3.706377)
- pair 3: +1.114882 (school -2.314315, check -3.429197)

### step130000

- pair 1: +1.055803 (school -2.732460, check -3.788263)
- pair 2: +0.830110 (school -2.974038, check -3.804148)
- pair 3: +1.095766 (school -2.209624, check -3.305390)

### step140000

- pair 1: +1.198607 (school -2.671318, check -3.869925)
- pair 2: +0.811851 (school -2.949193, check -3.761045)
- pair 3: +1.076843 (school -2.277194, check -3.354037)

### step143000

- pair 1: +1.366667 (school -2.464635, check -3.831302)
- pair 2: +0.765813 (school -3.129574, check -3.895387)
- pair 3: +1.023053 (school -2.261355, check -3.284408)
