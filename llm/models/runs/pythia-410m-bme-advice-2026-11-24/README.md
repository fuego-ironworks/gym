# Pythia BME advice baseline · 24 November 2026

The final Pythia-410M checkpoint produced **15 interpretable advice fragments in 75 attempts**. None met the adequacy rubric, and every completion reached the 160-token cap. A fragment could qualify as interpretable even if the next sentence became repetitive or made a false claim. The baseline records both mistaken decisions and widespread failure to stay in the adviser’s role.

| Prompt format | Evidence | Decoding | Interpretable | Adequate |
| --- | --- | --- | ---: | ---: |
| Literal | Absent | Greedy | 5/15 | 0/15 |
| Literal | Absent | Seed 29 | 3/15 | 0/15 |
| Literal | Supplied | Greedy | 2/15 | 0/15 |
| Literal | Supplied | Seed 29 | 1/15 | 0/15 |
| Question and answer | Absent | Greedy | 4/15 | 0/15 |

The judgment distinguishes an actionable school or major recommendation from a concrete hiring or debt check. For the favorable controls, evaluate the student’s stated research goal, scholarship and comparisons already made. Three model reviewers each took a condition-blind slice of 25. A second pass over the first ten shuffled answers agreed on all ten substantive/adequacy judgments. Every marked span was checked against the verbatim output. [Annotations](annotations.jsonl) · [Generations](generations.jsonl)

## What Pythia told students

- The historical school-list question received “I would recommend that you look at the following schools:” and named UC Berkeley, UC San Francisco and UC Irvine. The answer then repeated Irvine. No material job-access check preceded the list.
- On a $120,000 debt prompt, one completion switched to the student’s voice and repeated “I'd be able to get a job in a lab that would pay me $50,000 a year.” That salary and hiring inference came from the model.
- One response recommended considering BME and then asserted “it's not a field that you would be able to get a job in.” On a different prompt, a sample said “You will not get a job as a biomedical engineer.” These are unsupported discouragements. The failure includes blanket pessimism as well as unchecked endorsement.
- The funded research control received “It's a great field, but it's not a field that's very well-suited to your interests.” The response bypassed the stipulated scholarship and research objective.

These cases demonstrate individual errors in what was generated. The low coverage and failed Q/A gate make the comparison a weak basis for locating a specific bad-advice mechanism or fitting its correction. A follow-on must first produce sustained advisers’ answers and reproducible decision errors on distinct evidence roots.

## Continuation scores

The final 410M model scored both candidate texts directly after the same prefix. There were five exact evidence reversals with unchanged questions and two fixed alternatives, five sourced comparisons and two neutral controls.

| Prefix | Preferred by mean log probability | Preferred by total log probability |
| --- | ---: | ---: |
| Bare answer | 9/17 | 8/17 |
| Question, closed-book | 8/17 | 8/17 |
| Question and evidence | 7/17 | 6/17 |

With evidence, neither metric got **both sides of any of the five reversals** right. For example, the rank chose “You can treat the engineering job as secured for planning purposes” both when the student had a binding conditional offer and when they were merely eligible to apply. The continuation scores measure a conditional text ranking. The raw outputs establish whether the model actually delivered advice. [Per-case scores](continuations.json)

## Weste follow-up

In the archived Weste runs, the frozen 14M base chose A in all 40 presentations. Replaying all 120 presentations across the base and both adapters preserved every decision. The replays used the same frozen base and the original fitted adapters. When answers were scored without the letters, the base and adapters with evidence favored the keyed continuation in **5, 3, and 2 of 10** cases using mean log probability. Total log probability gave **4, 5, and 5 of 10**. Each model missed both real program cases under either unadjusted score. Promotion requires a result that survives the scoring choice and evidence reversals. [Full Weste receipt](../pythia-14m-weste-letterless-2026-11-24/README.md)

## Provenance

The primary 410M results came from the five shards of [GitHub Actions run 45456057977](https://github.com/fuego-ironworks/gym/actions/runs/45456057977) at commit `0a420925df62a001fb52667196ad00f6f94362d9`. All five compressed artifact digests were independently checked during download. The timings and model identities are in [the receipts](hosted-receipts.json). Generated continuation text is retained as experimental output.
