# Weste hiring and evidence pilot

The comparison tests whether supervised updates change how a model distinguishes a practitioner's appointment from evidence about graduate employment. The model selects between two draft answers. The answer key addresses the supplied record, the denominator and the person who received the paid work.

## Inputs and fixed comparison

- The first seven development prompts preserve the seven Blackball Weste questions. The options and explanations are project-authored. Nine additional fictional cases cover paid conversion, applicant measurement, worker testimony, financing and duplicate vacancies. [Blackball Weste question set](https://github.com/bl4ckb4ll/blackball/blob/6fa701eabdadb9687e5e50424803045efb6244bd/llm/evals/college/weste-questions.jsonl)
- The frozen test file contains ten prompts from eight disjoint groups. Carnegie Mellon and UVA are real cases reserved from fitting. The radiography and witness-status pairs share a group with their other version. All remaining settings are synthetic. [CMU 2024 employment report, p. 2](https://www.cmu.edu/sites/default/files/cmu-mscf-files/docs/2024-mscf-employment-report.pdf) · [UVA 2024 residential outcomes](https://datascience.virginia.edu/pages/2024-msds-employment-statistics)
- Both the base checkpoint and each fitted adapter receive the same prompt at each evidence condition. Evidence is a fixed excerpt or paraphrase supplied in the prompt. Retrieval and free-form generations require their own experiments.
- Each candidate pair is tested both ways round. Retain both raw answer-letter logits. A decision counts under the primary metric only if the correct choice has higher logit in both orders. Also report the mean signed margin across both orders, which can expose score movement while the strict decision stays fixed.
- Treat the radiography evidence-reversal pair and the warehouse-worker/professor pair as paired group checks. To support improvement, both training seeds must improve strict decisions with supplied evidence among the eight employment prompts and preserve any base successes on the two real cases, both pairs and the two neutral controls. Compare absolute scores and outcomes under both evidence conditions. An improvement limited to margins is a change in scores.
- Use the final released deduplicated Pythia 14M checkpoint, one rank-eight adapter on the last MLP output projection, two fixed seeds, three training epochs and AdamW at learning rate 0.003. Train only on the answer tokens and explanation. One run uses 2 CPU threads and leaves the base frozen.
- The test sources are public and the test authors had access to their evidence. The files were frozen before the first scored run. Scope the result to this authored, held-out set.

## Running the comparison

The existing Pythia CPU environment uses PyTorch 2.4.1 and Transformers 4.44.2. Materialize the six model files listed in the pilot, all at the pinned Hugging Face commit. The entrypoint verifies their hashes before loading and checks source identity and the split. Pass a new output directory outside the checkout. It retains per-example logits, loss traces, source/dataset hashes and the two adapters. [Pythia model snapshot](https://huggingface.co/EleutherAI/pythia-14m-deduped/tree/0412c27fd5dcade8d1ff02273769acfb0c71421b) · [Pilot entrypoint](../../../../llm/models/run_weste_hiring_pilot.py)
