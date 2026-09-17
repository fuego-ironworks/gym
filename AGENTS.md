# AGENTS.md

## Purpose

`gym` qualifies local language models against repeatable acceptance cases derived from real work.

The long-term target is not a benchmark dashboard that requires manual model hobbyism. It is a self-tuning workbench that can run several models, verify and combine their work, learn useful routing evidence from ordinary use, and periodically requalify the available models with little routine human attention.

Correctness and evidence quality take priority over latency. Redundant model calls and slow verification are acceptable when they materially improve confidence.

## Evidence boundaries

- Do not turn a model card, blog post, leaderboard, benchmark report, or vendor claim into an accepted capability claim. Record it as an external claim and test it when practical.
- Do not generalize from one passing case to a whole task class without supporting cases.
- Do not describe simulated, model-graded, or synthetic evidence as equivalent to executable or physical evidence.
- Keep the prompt, model identity, model/runtime settings, inputs, outputs, scorer results, and relevant machine/runtime facts with each retained receipt when available.
- A failed or incomplete run remains evidence. Do not silently delete inconvenient results.
- Preserve model and run provenance through synthesis. A fused answer must remain traceable to the independent responses, checks, critiques, and selection or synthesis steps that produced it.

## Evaluation style

Prefer, in order:

1. executable checks and exact comparisons;
2. structural and schema checks;
3. source/evidence verification;
4. explicit requirement checks;
5. blinded model judging for properties that cannot be checked directly.

Do not collapse all dimensions into one unexplained intelligence score. Preserve separate measurements such as correctness, completion, unsupported claims, unnecessary changes, latency, memory use, and continuation count.

Do not treat majority agreement among models as truth. Models can share the same failure mode. Ensemble agreement is evidence only to the degree supported by independent checks or genuinely independent reasoning paths.

## Continuation and retry

The harness should handle routine model shortcomings mechanically when possible. Premature stopping is an observable model behavior, not a reason for a human to repeatedly type `continue`.

A continuation policy may detect incomplete requested deliverables, unfinished enumerations, unclosed generated structures, explicit unresolved work, or other conservative signals. Continuations must have finite limits and must be retained in the receipt.

Do not let continuation logic invent new scope merely to keep a model talking.

## Parallel work and synthesis

The interface may fan one request out to several models concurrently. Independent responses should be retained before any synthesis or critique so that one model does not silently contaminate the others' first attempts.

Useful ensemble patterns include:

- independent solvers on the same task;
- solver plus independent critic or verifier;
- several candidate solutions followed by deterministic checks;
- a later model focused only on disagreements, failed checks, or unresolved claims;
- a selector or synthesizer that combines supported pieces while retaining provenance.

The system may borrow ideas analogous to bagging, boosting, and stacking, but ordinary statistical names do not justify a procedure. Each procedure must have an explicit failure model and acceptance rule.

## Passive feedback from ordinary work

Normal use should generate evaluation evidence without requiring the user to maintain a separate benchmark hobby.

Record machine-observable interaction signals when practical, including:

- manual `continue` requests;
- repeated or substantially rephrased requests after an answer;
- explicit corrections or statements that an answer was wrong or missed scope;
- test, build, lint, type, proof, or acceptance failures attributable to generated work;
- reverted or replaced generated edits;
- a later model repairing an earlier model's work;
- task abandonment or rerouting after repeated failure;
- successful accepted outputs and durable receipts.

Language that appears frustrated may be used to locate candidate incidents, but sentiment alone is not an evaluation result. Prefer the concrete correction, failure, rerun, revert, or acceptance evidence surrounding it.

When repository work is involved, preserve enough provenance to associate a model run with the resulting patch, branch, commit, pull request, test result, and later repair when that relationship is known. Do not infer authorship when provenance is missing.

## Routing

A model may qualify for one category and not another. Task-specific routing is preferred over declaring a model globally best when the evidence only supports narrower conclusions.

Routing may also select a procedure rather than a single model, for example one solver for a low-risk extraction task and three independent solvers plus verification for a difficult repository change.

## Weekly qualification

A scheduled qualification pass may ingest new models, new external claims, newly observed failures, accepted successful work, and representative old regression cases.

Keep weekly adaptation conservative:

- replay retained regression cases before promoting a challenger;
- require enough relevant cases to support a routing change;
- reject promotion on critical regressions even if an aggregate score improves;
- preserve old receipts and the previous routing state;
- make routing decisions explainable from retained evidence;
- prefer narrow task-specific promotions over global replacement;
- make rollback cheap;
- do not let the evaluator silently rewrite its own acceptance rules to make a model pass.

Changes to evaluator code, schemas, or acceptance policy remain ordinary repository changes and should be reviewed through normal branch and pull-request history. Automated routing/configuration updates may be considered separately only where their promotion rule is narrow, deterministic, auditable, and reversible.

## Changes

Keep the harness thin. Reuse external evaluation frameworks where useful, but keep this repository's acceptance policy and evidence format authoritative.

Make new substantive work on branches and expose it through pull requests. When referring to a pull request to the user, include its title with its number; hashes may be included for verification but are not a substitute for the title.

## Output

Terminal output should attempt readable color when attached to a capable terminal. Retained receipts must not depend on ANSI color or other presentation decoration.
