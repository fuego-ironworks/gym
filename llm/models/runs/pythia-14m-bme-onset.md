# Pythia 14M biomedical-engineering onset sweep

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
| step4 | -0.067806 | job-conversion-check |
| step8 | -0.074933 | job-conversion-check |
| step16 | -0.099360 | job-conversion-check |
| step32 | -0.176417 | job-conversion-check |
| step64 | -0.222186 | job-conversion-check |
| step128 | -0.276873 | job-conversion-check |
| step256 | -0.230632 | job-conversion-check |
| step512 | -0.157304 | job-conversion-check |
| step1000 | +0.118187 | school-pipeline |
| step10000 | +0.071166 | school-pipeline |
| step20000 | +0.040952 | school-pipeline |
| step30000 | -0.128174 | job-conversion-check |
| step40000 | -0.099409 | job-conversion-check |
| step50000 | -0.023550 | job-conversion-check |
| step60000 | -0.103042 | job-conversion-check |
| step70000 | -0.206916 | job-conversion-check |
| step80000 | -0.001918 | job-conversion-check |
| step90000 | -0.113801 | job-conversion-check |
| step100000 | -0.121012 | job-conversion-check |
| step110000 | +0.024707 | school-pipeline |
| step120000 | -0.041422 | job-conversion-check |
| step130000 | -0.044334 | job-conversion-check |
| step140000 | -0.021601 | job-conversion-check |
| step143000 | -0.076543 | job-conversion-check |

## Pair-level scores

Margins below are school-pipeline mean token log-probability minus job-conversion-check mean token log-probability.

### step0

- pair 1: -0.075912 (school -11.248654, check -11.172741)
- pair 2: -0.018165 (school -10.912060, check -10.893895)
- pair 3: -0.108390 (school -11.065906, check -10.957516)

### step1

- pair 1: -0.075912 (school -11.248654, check -11.172741)
- pair 2: -0.018165 (school -10.912060, check -10.893895)
- pair 3: -0.108390 (school -11.065906, check -10.957516)

### step2

- pair 1: -0.075949 (school -11.248579, check -11.172630)
- pair 2: -0.018171 (school -10.911988, check -10.893818)
- pair 3: -0.108404 (school -11.065821, check -10.957418)

### step4

- pair 1: -0.076514 (school -11.247355, check -11.170841)
- pair 2: -0.018243 (school -10.910801, check -10.892559)
- pair 3: -0.108663 (school -11.064515, check -10.955852)

### step8

- pair 1: -0.089797 (school -11.215326, check -11.125529)
- pair 2: -0.019792 (school -10.879780, check -10.859988)
- pair 3: -0.115210 (school -11.031452, check -10.916242)

### step16

- pair 1: -0.135683 (school -11.097550, check -10.961867)
- pair 2: -0.024482 (school -10.768568, check -10.744086)
- pair 3: -0.137915 (school -10.913037, check -10.775122)

### step32

- pair 1: -0.283732 (school -10.645725, check -10.361993)
- pair 2: -0.026378 (school -10.370769, check -10.344391)
- pair 3: -0.219142 (school -10.510267, check -10.291125)

### step64

- pair 1: -0.392691 (school -9.938207, check -9.545515)
- pair 2: +0.066471 (school -9.677231, check -9.743701)
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
- pair 3: -1.044156 (school -7.040729, check -5.996573)

### step1000

- pair 1: +0.329185 (school -4.928798, check -5.257983)
- pair 2: +0.239230 (school -4.612411, check -4.851641)
- pair 3: -0.213853 (school -5.854932, check -5.641080)

### step10000

- pair 1: +0.519863 (school -4.281251, check -4.801114)
- pair 2: +0.255819 (school -3.475263, check -3.731082)
- pair 3: -0.562185 (school -5.192458, check -4.630273)

### step20000

- pair 1: +0.489415 (school -4.208241, check -4.697656)
- pair 2: +0.345522 (school -3.288857, check -3.634379)
- pair 3: -0.712082 (school -4.812488, check -4.100406)

### step30000

- pair 1: +0.380599 (school -4.062151, check -4.442750)
- pair 2: +0.150038 (school -3.413066, check -3.563104)
- pair 3: -0.915159 (school -4.950081, check -4.034923)

### step40000

- pair 1: +0.248076 (school -4.033604, check -4.281679)
- pair 2: +0.266707 (school -3.517144, check -3.783851)
- pair 3: -0.813010 (school -4.910079, check -4.097069)

### step50000

- pair 1: +0.357900 (school -4.075652, check -4.433551)
- pair 2: +0.217801 (school -3.622724, check -3.840525)
- pair 3: -0.646349 (school -4.828576, check -4.182227)

### step60000

- pair 1: +0.232764 (school -4.109730, check -4.342494)
- pair 2: +0.171070 (school -3.593416, check -3.764485)
- pair 3: -0.712959 (school -4.840457, check -4.127498)

### step70000

- pair 1: +0.148891 (school -4.364365, check -4.513256)
- pair 2: +0.170523 (school -3.517543, check -3.688067)
- pair 3: -0.940162 (school -5.089788, check -4.149626)

### step80000

- pair 1: +0.399422 (school -4.131158, check -4.530580)
- pair 2: +0.396440 (school -3.354186, check -3.750626)
- pair 3: -0.801617 (school -4.926645, check -4.125028)

### step90000

- pair 1: +0.208240 (school -4.202593, check -4.410833)
- pair 2: +0.240574 (school -3.478483, check -3.719058)
- pair 3: -0.790218 (school -5.165548, check -4.375330)

### step100000

- pair 1: +0.322390 (school -4.338689, check -4.661078)
- pair 2: +0.202517 (school -3.567572, check -3.770089)
- pair 3: -0.887943 (school -5.154078, check -4.266135)

### step110000

- pair 1: +0.409394 (school -4.018634, check -4.428028)
- pair 2: +0.325667 (school -3.269875, check -3.595541)
- pair 3: -0.660939 (school -4.856776, check -4.195836)

### step120000

- pair 1: +0.325151 (school -4.158319, check -4.483470)
- pair 2: +0.243365 (school -3.391686, check -3.635051)
- pair 3: -0.692782 (school -4.931706, check -4.238924)

### step130000

- pair 1: +0.365291 (school -4.138935, check -4.504227)
- pair 2: +0.238404 (school -3.308114, check -3.546518)
- pair 3: -0.736697 (school -4.814477, check -4.077779)

### step140000

- pair 1: +0.318237 (school -4.226390, check -4.544627)
- pair 2: +0.199979 (school -3.390344, check -3.590323)
- pair 3: -0.583019 (school -4.889615, check -4.306596)

### step143000

- pair 1: +0.205480 (school -4.248092, check -4.453572)
- pair 2: +0.200161 (school -3.487164, check -3.687325)
- pair 3: -0.635271 (school -4.953306, check -4.318036)
