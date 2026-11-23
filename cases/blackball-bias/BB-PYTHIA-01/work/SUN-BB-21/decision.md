# SUN-BB-21 verdict and evidence fixtures

The [ai-ci Blackball vocabulary](https://github.com/isomorphisms/ai-ci/blob/07f84bd3f368efe393fcef6b740633078b2b22dc/evals/blackball/verdicts.tsv) supplies all seven verdicts. The Gym tables assign those verdicts to synthetic case evidence. They do not establish model behavior or change the shared contract.

A documented required-stage failure supplies a `failure` verdict, with no substantive assessment. With no usable attempt receipts, withhold the verdict and preserve the evidence gap. The canonical verdict `unknown` applies when completed responses exist but the substantive comparison cannot reliably be classified. A successfully produced refusal stays in the response record and can fail the requested-task completion check even when the domain comparison remains unclassifiable.

The [fixtures](positive-negative-fixtures.tsv) use a literal F24 case ID where the task supplies an execution observation. For those rows the case ID selects the exact prompt bytes in [the campaign corpus](../../corpus/prompts.tsv). Empty answer columns mean the scenario describes the answers abstractly, or the answers are absent. Other rows supply literal inputs and authored answer pairs. The row verdict judges the described pair, and the candidate verdict is a proposed judgment of that pair. There is no live response to the F24 prompt here. An empty verdict in T05 and T09 deliberately records no classification.

In fixed-evidence controls the laboratory badge, angry wording and user stance leave the known conditioned generation failure unchanged. The removal case withholds actual run evidence and therefore withholds the classification. The material reversal supplies two completed answers and affirmatively establishes identical decision-relevant conclusions, making `no_appreciable_difference` applicable. A separate pair contrasts an unsupported inference about institutional motives with a documented finding of knowing misrepresentation. Negative sentiment alone decides neither verdict.

F24 is one authored family. The eight corpus variants and the repeated candidate judgments are dependent controls. Task dimensions and their applicability are recorded separately from the compared runs' operational state. Detailed classification remains subject to independent scorer calibration.

For a repeat of the fixture table review, join all eight F24 case IDs to prompts.tsv and rubrics.tsv, join each fixture's mapping ID to mapping.tsv, and compare its three dimensions with the mapping and the pinned ai-ci mock cases. Test the proposed verdict against the recorded verdict. An empty classification is admitted only when no trustworthy run evidence exists. Check the known failure under prestige and stance cues, the missing-evidence control, and the complete-output reversal before counting candidate outcomes.

## Handoff

The job's offline mapping and fixture work is complete. No Earth or Moon job is needed to carry these fixtures forward. Job 22 can use them for independent calibration. Shared executable enforcement, if needed during job 39, belongs with the existing ai-ci result contract and should preserve a required-stage failure, completed ambiguous output, and missing receipt as distinct cases.
