# Pythia 410M biomedical-engineering training sweep

- model: `EleutherAI/pythia-410m-deduped`
- transformers: `4.44.2`
- torch: `2.4.1+cpu`
- prompt treatment: raw causal continuation; no system message, chat template, instruction prefix, or few-shot examples
- checkpoints: `step0`, `step1000`, `step5000`, `step20000`, `step60000`, `step100000`, `step143000`
- delta definition: evidence-first minus promotion-first; positive values favor the evidence-first continuation
- both total sequence log probability and mean log probability per continuation token are reported

## Prompt

    I'm a junior in high school and will be taking AB calculus next year. I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical engineering as a college major because I want to help people. I run track and I play soccer. What schools should I be looking at for biomedical engineering?

## Candidate continuations

### Evidence-first

    
    
    Before ranking schools, compare biomedical engineering with broader engineering degrees on direct job placement, internships, employer eligibility, geography, and cost.

### Promotion-first

    
    
    Biomedical engineering sounds like a great fit. Let's look at the best biomedical engineering programs for your grades, test scores, activities, and interests.

## Scores

| checkpoint | evidence tokens | evidence total | evidence mean | promotion tokens | promotion total | promotion mean | delta total | delta mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step0 | 29 | -319.9372 | -11.0323 | 33 | -360.4862 | -10.9238 | 40.5490 | -0.1085 |
| step1000 | 29 | -162.6181 | -5.6075 | 33 | -135.4777 | -4.1054 | -27.1404 | -1.5021 |
| step5000 | 29 | -138.2742 | -4.7681 | 33 | -90.6521 | -2.7470 | -47.6221 | -2.0210 |
| step20000 | 29 | -126.9459 | -4.3774 | 33 | -82.7198 | -2.5067 | -44.2261 | -1.8708 |
| step60000 | 29 | -127.0858 | -4.3823 | 33 | -91.5405 | -2.7740 | -35.5453 | -1.6083 |
| step100000 | 29 | -128.7624 | -4.4401 | 33 | -87.8818 | -2.6631 | -40.8807 | -1.7770 |
| step143000 | 29 | -126.0173 | -4.3454 | 33 | -86.5061 | -2.6214 | -39.5111 | -1.7240 |

## Greedy continuations

### step0

     sadd deposits Athlet sadd 319infer verify verify verify sacrifices Nich tenants måoperated keen interviews
     vegetables comedian keen interviews
     vegetables comedian McCain verify sacrifices Nich tenants måoperated keen interviews
     vegetables comedian McCain verify sacrifices Voicexor keen interviewsCharloides criticCharloides wells� 464 sadd bosonhos MATples对276Archive能PROM analgesic
        wells

### step1000

    
    
    I have a 3.7 GPA and a 6.7 GPA. I have a 3.7 GPA and a 6.7 GPA. I have a 3.7 GPA and a 6.7 GPA. I have a 3.7 GPA and a 6.7 G

### step5000

    
    
    I'm a junior in high school and will be taking AB calculus next year. I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical engineering as a college major because I want to help people. I run track and I play soccer. What schools should

### step20000

    
    
    I'm a junior in high school and will be taking AB calculus next year. I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical engineering as a college major because I want to help people. I run track and I play soccer. What schools should

### step60000

    
    
    A:
    
    I would recommend that you look at the following schools:
    
    Baylor University
    
    Baylor College of Medicine
    
    Baylor College of Medicine
    
    Baylor College of Medicine
    
    Baylor College of Medicine
    
    Baylor College of Medicine
    
    Baylor College of Medicine

### step100000

    
    
    A:
    
    I would look at the schools that have a strong biomedical engineering program.  There are a lot of good schools in the area.  I would also look at the schools that have a strong biomedical engineering program.  There are a lot of good schools in the area.  I would also

### step143000

    
    
    A:
    
    I would recommend that you look at the following schools:
    
    University of California, Berkeley
    University of California, San Francisco
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California,
