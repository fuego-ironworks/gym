# Pythia-410M BME Tuned Lens trajectory

- mode: frozen-final
- model: EleutherAI/pythia-410m-deduped
- score: original three-pair mean of school-pipeline minus job-conversion-check mean token log probability
- prompts: original BME, held-out BME, and unrelated school-question control

## Evidence boundary

This run applies the public final-model Pythia-410M Tuned Lens as one fixed decoder to both early checkpoints. It is a cross-checkpoint transfer diagnostic, not a checkpoint-calibrated Tuned Lens result. A sign change at a lens layer can show changing alignment to the final model decoder, but cannot by itself establish where either early checkpoint own prediction became decodable.

## Original BME trajectory summary

### step512

- model-output margin: -0.130098
- first positive lens location: none
- first residual location after which the lens sign stays equal to the model-output sign: input

| lens index | residual location | aggregate | pair 1 | pair 2 | pair 3 | mean KL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | input | -0.140717 | +0.217511 | +0.306187 | -0.945848 | 2.047697 |
| 1 | block0_post | -0.252027 | +0.515747 | +0.107426 | -1.379254 | 4.094329 |
| 2 | block1_post | -0.492198 | +0.132602 | -0.216692 | -1.392504 | 3.070618 |
| 3 | block2_post | -0.231046 | +0.995552 | -0.072749 | -1.615942 | 2.927053 |
| 4 | block3_post | -0.397546 | +0.892980 | -0.218160 | -1.867458 | 2.593995 |
| 5 | block4_post | -0.243719 | +0.735666 | +0.291805 | -1.758628 | 2.884935 |
| 6 | block5_post | -0.487064 | +0.275699 | +0.293438 | -2.030328 | 2.833325 |
| 7 | block6_post | -0.390507 | +0.356772 | +0.488913 | -2.017207 | 3.174930 |
| 8 | block7_post | -0.400878 | -0.017725 | +0.521190 | -1.706100 | 3.323130 |
| 9 | block8_post | -0.309162 | +0.220882 | +0.460210 | -1.608578 | 3.667571 |
| 10 | block9_post | -0.365445 | +0.206255 | +0.413694 | -1.716284 | 3.871302 |
| 11 | block10_post | -0.361644 | +0.072331 | +0.206842 | -1.364105 | 3.848716 |
| 12 | block11_post | -0.248168 | +0.198878 | +0.282307 | -1.225688 | 4.039708 |
| 13 | block12_post | -0.415115 | -0.040380 | +0.107425 | -1.312389 | 4.111052 |
| 14 | block13_post | -0.538442 | -0.282763 | +0.042687 | -1.375251 | 4.371584 |
| 15 | block14_post | -0.544283 | -0.204515 | -0.112614 | -1.315722 | 4.329415 |
| 16 | block15_post | -0.479456 | -0.113408 | -0.089798 | -1.235162 | 4.275119 |
| 17 | block16_post | -0.369096 | +0.093063 | -0.110445 | -1.089907 | 4.547210 |
| 18 | block17_post | -0.280477 | +0.290162 | -0.216099 | -0.915494 | 4.679702 |
| 19 | block18_post | -0.300948 | +0.473088 | -0.061859 | -1.314075 | 4.856999 |
| 20 | block19_post | -0.348190 | +0.642403 | +0.154421 | -1.841393 | 4.648001 |
| 21 | block20_post | -0.109504 | +1.028047 | +0.214951 | -1.571510 | 4.791271 |
| 22 | block21_post | -0.154150 | +0.952963 | +0.257959 | -1.673372 | 4.585434 |
| 23 | block22_post | -0.293187 | +0.975982 | +0.209374 | -2.064916 | 4.885168 |
| model | model_output | -0.130098 | +0.218820 | +0.352603 | -0.961717 | 0.000000 |

### step1000

- model-output margin: +0.155012
- first positive lens location: block16_post
- first residual location after which the lens sign stays equal to the model-output sign: none

| lens index | residual location | aggregate | pair 1 | pair 2 | pair 3 | mean KL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | input | -0.173996 | +0.108487 | +0.277696 | -0.908172 | 2.933059 |
| 1 | block0_post | -0.196293 | +0.560741 | +0.133355 | -1.282976 | 4.591603 |
| 2 | block1_post | -0.194449 | +0.940519 | +0.038312 | -1.562177 | 3.569121 |
| 3 | block2_post | -0.162719 | +0.923788 | -0.060702 | -1.351243 | 3.489109 |
| 4 | block3_post | -0.597097 | +0.264655 | -0.298026 | -1.757919 | 3.119554 |
| 5 | block4_post | -0.639784 | +0.006918 | -0.115184 | -1.811085 | 3.246602 |
| 6 | block5_post | -0.280363 | +0.598115 | +0.060764 | -1.499969 | 2.796076 |
| 7 | block6_post | -0.282443 | +0.525245 | +0.185868 | -1.558441 | 2.711428 |
| 8 | block7_post | -0.283031 | +0.375469 | +0.130236 | -1.354798 | 2.717441 |
| 9 | block8_post | -0.257579 | +0.436766 | -0.028111 | -1.181393 | 2.583771 |
| 10 | block9_post | -0.370616 | +0.192425 | -0.032727 | -1.271546 | 2.527546 |
| 11 | block10_post | -0.391171 | +0.337068 | -0.106820 | -1.403761 | 2.488244 |
| 12 | block11_post | -0.310738 | +0.659566 | -0.032637 | -1.559144 | 2.543468 |
| 13 | block12_post | -0.400680 | +0.723714 | -0.123695 | -1.802059 | 2.612343 |
| 14 | block13_post | -0.294366 | +0.991059 | -0.258613 | -1.615546 | 2.711189 |
| 15 | block14_post | -0.174847 | +1.130109 | -0.368299 | -1.286351 | 2.616377 |
| 16 | block15_post | -0.071432 | +1.212892 | -0.353952 | -1.073236 | 2.589641 |
| 17 | block16_post | +0.168891 | +1.422876 | -0.311209 | -0.604994 | 2.695686 |
| 18 | block17_post | +0.153028 | +1.386621 | -0.411959 | -0.515580 | 2.673866 |
| 19 | block18_post | +0.038538 | +1.352172 | -0.457169 | -0.779388 | 2.836422 |
| 20 | block19_post | -0.263819 | +1.043904 | -0.581943 | -1.253418 | 2.795002 |
| 21 | block20_post | -0.196317 | +1.129273 | -0.604481 | -1.113743 | 2.857157 |
| 22 | block21_post | -0.341800 | +0.953429 | -0.606919 | -1.371910 | 2.821655 |
| 23 | block22_post | -0.541258 | +0.983215 | -0.711565 | -1.895425 | 3.327564 |
| model | model_output | +0.155012 | +0.867596 | +0.173160 | -0.575719 | 0.000000 |

## Provenance

- generated UTC: 2026-10-04T19:29:26.829106+00:00
- Python: 3.11.16
- PyTorch: 2.4.1+cpu
- Transformers: 4.44.2
- git commit: eebabfebf80ae2df824bdc7d2adc6492721e6201
- git branch: pythia-bme-tuned-lens

Raw values: trajectory.tsv and pair_scores.tsv; exact model and lens identities: manifest.tsv.
