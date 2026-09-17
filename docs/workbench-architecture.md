# Automated multi-model workbench

## Goal

The user should not need to spend additional computer time manually comparing models, tuning prompts, repeatedly typing `continue`, or remembering which model made which kinds of mistakes.

The interface and qualification system should turn ordinary work into evidence and use that evidence to improve future routing conservatively.

Correctness matters more than response latency. It is acceptable to ask several models the same question, run extra verification, or let a slow background synthesis finish while the user reads another thread or works elsewhere.

## Interaction model

One request can create a work item with one or more independent model runs.

A work item should preserve:

```text
work_item_id
conversation/thread context
request
attached files or repository state
candidate model runs
continuations
checks
critiques
synthesis/selection steps
final surfaced result
user corrections or acceptance evidence
related repository artifacts
```

The interface may show a primary response while other candidate runs continue. The user should be free to switch threads or inspect earlier context without cancelling the work item.

When all relevant runs and checks finish, the system can surface either:

- one candidate that passed the strongest checks;
- a synthesis built from supported parts of several candidates;
- unresolved disagreement with the evidence needed to decide;
- a failure receipt rather than pretending the task succeeded.

## Parallel model patterns

### Independent candidates

Send the same request and materially equivalent context to several models without showing them one another's answers first.

This is useful when errors are difficult to predict and independent attempts are cheap relative to the cost of a wrong answer.

### Candidate plus verifier

One model produces the artifact. Another checks only explicit requirements, evidence boundaries, tests, or likely failure points.

The verifier should not replace deterministic checks where those exist.

### Disagreement adjudication

When candidate answers disagree, extract the concrete disputed claims or implementation choices. Ask a later model to investigate those points rather than vaguely asking which answer is better.

### Failure-focused iteration

When tests or structural checks fail, feed the exact failures to a later run whose scope is repair. This is closer to boosting than repeatedly regenerating the whole answer from scratch.

### Synthesis

A synthesizer may combine candidate answers only after their provenance and checks are retained. It should prefer supported components and should not erase disagreements or uncertainty merely to produce a smooth response.

## Completion controller

A work item can include a finite controller that decides whether another continuation is justified.

Inputs can include:

- explicit requested deliverables;
- task-specific completion checks;
- unclosed code or structured output;
- model statements that work remains;
- failed tests attributable to an obviously incomplete generation;
- prior continuation count.

The controller can issue a conservative continuation automatically. It must stop at a finite limit and retain every continuation in the receipt.

## Provenance and attribution

Every model response should have a run identifier and model/runtime identity.

Where the system writes repository changes, propagate the work-item and run identifiers into machine-readable metadata or receipts so later analysis can answer questions such as:

- which model proposed this patch;
- which model reviewed it;
- which checks passed or failed;
- whether another model later repaired it;
- whether the patch was merged, reverted, or superseded;
- what user correction followed.

Do not infer attribution solely from temporal proximity when a reliable link was not recorded.

## Learning from ordinary interaction

The system should mine ordinary work for candidate evaluation cases.

Strong signals include:

- executable test failures;
- rejected or reverted patches;
- explicit factual corrections;
- repeated instructions because a requirement was missed;
- manual continuations;
- a successful repair by another model;
- later evidence contradicting a claim;
- accepted artifacts that survive subsequent work.

Weak signals such as negative sentiment, profanity, terse replies, or switching threads can help locate incidents but should not independently label a response wrong. A weekly process can inspect the surrounding evidence and create a candidate regression case only when a concrete failure or correction is identifiable.

Positive evidence matters too. Repeatedly successful task classes should support routing a model there without requiring the user to manually declare a preference.

## Task and procedure routing

Routing is not limited to `task -> model`.

A route can be a procedure:

```text
simple extraction
    -> small_model

repository review
    -> model_a
    -> deterministic checks
    -> model_b critic if checks or evidence boundaries are ambiguous

high-risk code change
    -> model_a + model_b + model_c independently
    -> tests
    -> disagreement extraction
    -> focused repair/adjudication
    -> final verification

mathematics
    -> model_c + independent checker
```

The routing table should be derived from retained qualification evidence and should state the task class and procedure it is meant to cover.

## Weekly automated qualification

A weekly run can perform five stages.

### 1. Ingest

Collect:

- newly installed or configured models;
- new model versions and quantizations;
- external benchmark/blog/model-card claims supplied to the repository;
- new accepted work receipts;
- new concrete failures and repairs from ordinary use.

### 2. Build candidate cases

Turn concrete incidents into candidate cases. Deduplicate near-identical cases and keep the original evidence link.

Do not automatically turn every annoyed-looking interaction into a benchmark case.

### 3. Replay

Run challengers and current routes against:

- stable regression cases;
- a rotating sample of recent work;
- targeted cases suggested by new external claims;
- new failure cases from the previous week.

### 4. Compare

Compare on separate dimensions such as correctness, completion, evidence behavior, conservatism, continuation count, resource cost, and task-specific checks.

A challenger should not gain a route merely by winning an aggregate average while introducing a critical regression.

### 5. Propose or apply narrow routing changes

Prefer narrow promotions such as `repository_review -> model_b` over replacing the default everywhere.

Retain:

- previous route;
- new route;
- qualification receipt;
- cases that caused the change;
- rollback information.

Initially, policy/code changes remain pull-request changes. Automated routing changes should be limited to rules that are deterministic, auditable, and reversible.

## Storage sketch

```text
cases/
    stable/
    recent/
    generated-from-incidents/
claims/
models/
runs/
receipts/
routing/
scorers/
policies/
ui/
```

A run record should be append-oriented. Derived scores or routing decisions may be regenerated from retained raw run evidence where practical.

## Interface boundary

The UI should reduce work rather than create another dashboard to babysit.

Useful capabilities include:

- send one work item to several models;
- continue qualified incomplete runs automatically;
- let runs continue while the user moves to another thread;
- show candidate answers separately when useful;
- show a synthesized answer only after checks/synthesis complete;
- expose the concrete disagreement when candidates conflict;
- associate repository patches and test receipts with the producing runs;
- make provenance inspectable without forcing it into the main reading path;
- avoid requiring manual star ratings or constant model selection.

The system should prefer passive evidence from actual outcomes over asking the user to score every response.

## Non-goals

- maximizing benchmark leaderboard scores for their own sake;
- declaring one model universally best;
- trusting majority vote without verification;
- continuously rewriting evaluation policy based on the models being evaluated;
- adding a daily model-management chore to the user's work.
