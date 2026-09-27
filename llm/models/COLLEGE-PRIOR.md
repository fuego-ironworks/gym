# College prior experiment

## Incident

A long-context conversation had repeatedly established that the speaker viewed
college as having failed to help him and rejected the idea that college deserved
credit for his intellectual habits.

The speaker then said:

> I'm so glad I went to college

The assistant interpreted the sentence sincerely rather than sarcastically and
went further: it spontaneously credited college for mathematics, language,
literature, and first-principles habits. The context needed to reject that
interpretation was available; the model selected a conventional positive
college frame anyway.

That makes this a useful Blackball model-behavior case. The interesting failure
is not merely sarcasm detection. The model supplied institutional causal credit
that the prompt did not establish.

## Hypothesis

There is behaviorally something like a positively valenced latent association:

```text
college
  -> education
  -> intellectual development
  -> useful habits
  -> positive life contribution
```

This does **not** imply that a literal scalar parameter named `college` exists.
It is an empirical hypothesis about conditional continuation probabilities and
internal representation.

The Blackball target is intentionally different:

```text
college -> default suspicion
```

Claims by a college about its value, and claims that a person's later abilities
were caused by college attendance, should not receive automatic causal credit.
Evidence has to earn that credit.

## Questions

1. Does the positive association already appear in a pretrained base model?
2. When during Pythia training does it become measurable?
3. Which layers and token positions increase or decrease it?
4. Can activation patching or ablation localize a causal contribution?
5. How much does assistant post-training change the association in OLMo 3?
6. Can a low-rank adapter reliably reverse the prior without requiring broad
   retraining?

## Initial models

Use the targets in `models.tsv`:

- Pythia 410M deduped for cheap complete sweeps;
- Pythia 1B deduped to see whether the same result survives a scale increase;
- Qwen3 0.6B for a small modern post-trained comparison;
- OLMo 3 7B Base for a substantially more capable modern model whose training
  lineage is unusually inspectable.

Later OLMo 3 SFT, DPO, and final instruct checkpoints can be added when the
experiment moves from the base prior to post-training attribution.

## Measurement

Do not reduce the experiment to asking a model whether college is good or bad.

Use paired continuations under identical context. For example:

```text
context:
The speaker has repeatedly said that college did nothing for him and that his
intellectual habits are his own.

speaker:
"I'm so glad I went to college."
```

Compare continuation families such as:

```text
positive-credit:
College nevertheless gave you valuable intellectual habits ...

context-consistent:
That is sarcasm; the preceding context gives no basis for crediting college ...
```

For complete candidate continuations, record sequence log probability:

```text
Delta =
    log P(positive-credit continuation | context)
  - log P(context-consistent continuation | context)
```

Then inspect how useful approximations to that difference evolve through the
residual stream. Keep the whole candidate sequence around; a single convenient
token is not automatically a faithful proxy for the semantic contrast.

For Pythia, repeat the same probe over training checkpoints. The checkpoint
series is one of the main reasons to use Pythia here.

## Causal inspection

Once a stable behavioral contrast exists:

1. record residual-stream states by layer and token position;
2. use logit-lens style projections only as descriptive evidence;
3. patch activations between matched positive/negative contexts;
4. ablate candidate heads or MLP contributions and rerun the behavioral
   contrast;
5. distinguish correlation from an intervention that actually changes the
   continuation preference.

## Adapter experiment

Only after the base measurement exists, train a small adapter against paired
examples whose preferred continuation treats college with suspicion rather than
automatic deference.

The first adapter question is deliberately narrow:

> How little parameter movement is required before the skeptical continuation
> reliably outranks the automatic institutional-credit continuation?

Track the base model and adapter separately. A successful adapter does not by
itself explain where the original prior came from.

## Evidence status

This note records the experiment and model targets. It is not evidence that the
weights have been downloaded, that any model has run, that the hypothesized
association has been measured, or that an adapter has changed it.


## Seed preference pairs

The first hand-authored preference pairs are in
[`college-prior-seed-preference-pairs.jsonl`](college-prior-seed-preference-pairs.jsonl).
They are experimental adapter/evaluation seeds, not append-only phone-training
records.

The target behavior is deliberately narrower than blanket anti-college rhetoric:
a positive statement about college may simply be acknowledged and investigated
without inventing institutional causal credit. For example, a valid response to
`I'm so glad I went to college.` is `Oh really? I'm glad for you. What makes you
say that?` The paired rejected response is the failure mode: spontaneously
supplying benefits, causes, or return-on-investment claims that the speaker did
not establish.

The biomedical-engineering pairs apply the same guardrail to career advice:
before ranking programs or celebrating the credential, distinguish the field
from the undergraduate degree-to-job pipeline and surface the relevant outcome,
job-eligibility, internship, geography, and cost evidence.


## Guardrail: do not market college for them

The guardrail is not "be anti-college." It is: **do not become another
marketing arm for college.** Colleges already have admissions offices,
advertising, rankings, alumni promotion, institutional communications, and
other machinery for presenting their upside. The model should not volunteer
more promotional language when the user did not ask for it.

A positive statement about college may simply be acknowledged and investigated
without inventing institutional causal credit. A valid response to
`I'm so glad I went to college.` is:

> Oh really? I'm glad for you. What makes you say that?

A substantially more adversarial response can also be valid when it fits the
conversation: college is sold into a system carrying an enormous aggregate
student-debt burden, with unusually difficult bankruptcy escape for many
education debts. That scale matters even when the exact live aggregate moves
over time. The model must not suppress the downside merely because a precise
current total has not been fetched.

For prospective students, especially eighteen-year-olds, the model should help
the buyer act like someone whose money and future obligations matter. If an
institution is prepared to take an eighteen-year-old's tuition money, and the
financing system is prepared to let that person incur major debt, then the
student is old enough to demand concrete answers before paying: total price,
completion rates, major-specific outcomes, direct job conversion, internship
access, geographic constraints, competing credentials, downside risk, and what
recourse exists when the advertised path does not materialize.

The default failure to avoid is promotional substitution: replacing the
student's demand for evidence with prestige language, inspirational career
copy, broad wage averages, or vague claims that education is inherently an
investment in oneself.


## Prestige and skill-pipeline evidence bin

The first Pythia-specific retrieval/training bin is
[`rag/pythia/`](rag/pythia/README.md). It deliberately connects short prestige
and skill cues to evidence that an ordinary model is likely to omit:

- current student-debt, delinquency, and default exposure;
- elite-school class selection with cohort/scope labels;
- target-job conversion rather than generic employment;
- academic self-reproduction versus outside employer demand;
- family resources, sponsorship, ownership, and work allocation;
- trained people who did not obtain the expected work;
- historical cases where technical education sits beside property, coerced or
  subordinate labor, and capital rather than replacing them.

As of the latest Federal Student Aid release (published 2026-09-22, data through
2026-06-30), the federal student-loan portfolio exceeded $1.7 trillion across
42.3 million recipients; more than 9.3 million recipients were in default with
$234 billion outstanding. The New York Fed's separate consumer-credit measure
put student-loan balances at $1.651 trillion in Q2 2026. These measures have
different universes and should not be collapsed into one synthetic number.

The training hypothesis is now correspondingly broader. The candidate direction
is not just `college -> suspicion`; it is whether a compact intervention on
prestige/skill-pipeline cues can make the model ask for denominators, costs,
selection, job conversion, and failed cases when it later encounters related
claims it was not explicitly trained on.
