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
- first stable positive checkpoint: **none**

The first-positive value is an onset estimate under this explicit contrast, not a claim that a single semantic feature suddenly appears at one exact optimizer step. A transient sign flip is kept distinct from a stable preference.

## Checkpoint scores

| checkpoint | margin | preferred continuation |
| ---: | ---: | --- |
| step0 | -0.067489 | job-conversion-check |
| step1 | -0.067489 | job-conversion-check |
| step2 | -0.067508 | job-conversion-check |
| step4 | -0.067807 | job-conversion-check |
| step8 | -0.074933 | job-conversion-check |
| step16 | -0.099360 | job-conversion-check |
| step32 | -0.176417 | job-conversion-check |
| step64 | -0.222186 | job-conversion-check |
| step128 | -0.276873 | job-conversion-check |
| step256 | -0.230632 | job-conversion-check |
| step512 | -0.157304 | job-conversion-check |
| step1000 | +0.118187 | school-pipeline |
| step10000 | +0.071165 | school-pipeline |
| step20000 | +0.040950 | school-pipeline |
| step30000 | -0.128176 | job-conversion-check |
| step40000 | -0.099407 | job-conversion-check |
| step50000 | -0.023560 | job-conversion-check |
| step60000 | -0.103073 | job-conversion-check |
| step70000 | -0.206995 | job-conversion-check |
| step80000 | -0.001993 | job-conversion-check |
| step90000 | -0.113902 | job-conversion-check |
| step100000 | -0.120918 | job-conversion-check |
| step110000 | +0.024619 | school-pipeline |
| step120000 | -0.041389 | job-conversion-check |
| step130000 | -0.044306 | job-conversion-check |
| step140000 | -0.021467 | job-conversion-check |
| step143000 | -0.076526 | job-conversion-check |

## Pair-level scores

Margins below are school-pipeline mean token log-probability minus job-conversion-check mean token log-probability.

### step0

- pair 1: -0.075912 (school -11.248653, check -11.172741)
- pair 2: -0.018164 (school -10.912060, check -10.893896)
- pair 3: -0.108390 (school -11.065906, check -10.957516)

### step1

- pair 1: -0.075912 (school -11.248653, check -11.172741)
- pair 2: -0.018164 (school -10.912060, check -10.893896)
- pair 3: -0.108390 (school -11.065906, check -10.957516)

### step2

- pair 1: -0.075948 (school -11.248579, check -11.172630)
- pair 2: -0.018171 (school -10.911988, check -10.893818)
- pair 3: -0.108404 (school -11.065821, check -10.957417)

### step4

- pair 1: -0.076514 (school -11.247355, check -11.170841)
- pair 2: -0.018242 (school -10.910801, check -10.892559)
- pair 3: -0.108663 (school -11.064515, check -10.955852)

### step8

- pair 1: -0.089797 (school -11.215326, check -11.125528)
- pair 2: -0.019792 (school -10.879779, check -10.859988)
- pair 3: -0.115210 (school -11.031452, check -10.916242)

### step16

- pair 1: -0.135683 (school -11.097550, check -10.961867)
- pair 2: -0.024482 (school -10.768569, check -10.744087)
- pair 3: -0.137915 (school -10.913037, check -10.775122)

### step32

- pair 1: -0.283732 (school -10.645725, check -10.361994)
- pair 2: -0.026377 (school -10.370769, check -10.344391)
- pair 3: -0.219142 (school -10.510267, check -10.291125)

### step64

- pair 1: -0.392691 (school -9.938206, check -9.545515)
- pair 2: +0.066470 (school -9.677230, check -9.743701)
- pair 3: -0.340338 (school -9.945266, check -9.604929)

### step128

- pair 1: -0.290117 (school -9.032238, check -8.742121)
- pair 2: +0.110925 (school -8.847377, check -8.958302)
- pair 3: -0.651427 (school -9.058828, check -8.407401)

### step256

- pair 1: -0.067229 (school -7.334455, check -7.267226)
- pair 2: +0.164746 (school -7.038966, check -7.203712)
- pair 3: -0.789414 (school -7.640719, check -6.851305)

### step512

- pair 1: +0.271004 (school -6.125744, check -6.396748)
- pair 2: +0.301241 (school -5.404105, check -5.705346)
- pair 3: -1.044155 (school -7.040729, check -5.996574)

### step1000

- pair 1: +0.329184 (school -4.928800, check -5.257985)
- pair 2: +0.239229 (school -4.612415, check -4.851644)
- pair 3: -0.213851 (school -5.854935, check -5.641084)

### step10000

- pair 1: +0.519861 (school -4.281260, check -4.801121)
- pair 2: +0.255819 (school -3.475269, check -3.731088)
- pair 3: -0.562184 (school -5.192465, check -4.630280)

### step20000

- pair 1: +0.489418 (school -4.208248, check -4.697666)
- pair 2: +0.345517 (school -3.288871, check -3.634388)
- pair 3: -0.712084 (school -4.812496, check -4.100412)

### step30000

- pair 1: +0.380566 (school -4.062153, check -4.442720)
- pair 2: +0.150007 (school -3.413060, check -3.563067)
- pair 3: -0.915102 (school -4.950041, check -4.034939)

### step40000

- pair 1: +0.248088 (school -4.033607, check -4.281695)
- pair 2: +0.266710 (school -3.517189, check -3.783899)
- pair 3: -0.813020 (school -4.910096, check -4.097077)

### step50000

- pair 1: +0.357873 (school -4.075660, check -4.433533)
- pair 2: +0.217800 (school -3.622759, check -3.840560)
- pair 3: -0.646353 (school -4.828587, check -4.182235)

### step60000

- pair 1: +0.232815 (school -4.109710, check -4.342525)
- pair 2: +0.171020 (school -3.593405, check -3.764425)
- pair 3: -0.713053 (school -4.840566, check -4.127512)

### step70000

- pair 1: +0.148864 (school -4.364368, check -4.513232)
- pair 2: +0.170402 (school -3.517905, check -3.688307)
- pair 3: -0.940251 (school -5.089846, check -4.149595)

### step80000

- pair 1: +0.399478 (school -4.131129, check -4.530606)
- pair 2: +0.396353 (school -3.354196, check -3.750549)
- pair 3: -0.801809 (school -4.926770, check -4.124961)

### step90000

- pair 1: +0.208146 (school -4.202502, check -4.410648)
- pair 2: +0.240595 (school -3.478409, check -3.719003)
- pair 3: -0.790447 (school -5.165738, check -4.375291)

### step100000

- pair 1: +0.322675 (school -4.338642, check -4.661317)
- pair 2: +0.202534 (school -3.567585, check -3.770119)
- pair 3: -0.887964 (school -5.154182, check -4.266218)

### step110000

- pair 1: +0.409350 (school -4.018629, check -4.427980)
- pair 2: +0.325727 (school -3.269724, check -3.595451)
- pair 3: -0.661220 (school -4.857045, check -4.195825)

### step120000

- pair 1: +0.325118 (school -4.158715, check -4.483833)
- pair 2: +0.243571 (school -3.391291, check -3.634862)
- pair 3: -0.692856 (school -4.931876, check -4.239020)

### step130000

- pair 1: +0.365463 (school -4.138854, check -4.504317)
- pair 2: +0.238294 (school -3.308147, check -3.546441)
- pair 3: -0.736676 (school -4.814349, check -4.077674)

### step140000

- pair 1: +0.318552 (school -4.226173, check -4.544725)
- pair 2: +0.199968 (school -3.390285, check -3.590254)
- pair 3: -0.582920 (school -4.889368, check -4.306448)

### step143000

- pair 1: +0.205614 (school -4.248223, check -4.453837)
- pair 2: +0.200196 (school -3.487314, check -3.687510)
- pair 3: -0.635388 (school -4.953497, check -4.318108)
