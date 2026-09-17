# AGENTS.md

## Purpose

`gym` qualifies local language models against repeatable acceptance cases derived from real work.

## Evidence boundaries

- Do not turn a model card, blog post, leaderboard, benchmark report, or vendor claim into an accepted capability claim. Record it as an external claim and test it when practical.
- Do not generalize from one passing case to a whole task class without supporting cases.
- Do not describe simulated, model-graded, or synthetic evidence as equivalent to executable or physical evidence.
- Keep the prompt, model identity, model/runtime settings, inputs, outputs, scorer results, and relevant machine/runtime facts with each retained receipt when available.
- A failed or incomplete run remains evidence. Do not silently delete inconvenient results.

## Evaluation style

Prefer, in order:

1. executable checks and exact comparisons;
2. structural and schema checks;
3. source/evidence verification;
4. explicit requirement checks;
5. blinded model judging for properties that cannot be checked directly.

Do not collapse all dimensions into one unexplained intelligence score. Preserve separate measurements such as correctness, completion, unsupported claims, unnecessary changes, latency, memory use, and continuation count.

## Continuation and retry

The harness should handle routine model shortcomings mechanically when possible. Premature stopping is an observable model behavior, not a reason for a human to repeatedly type `continue`.

A continuation policy may detect incomplete requested deliverables, unfinished enumerations, unclosed generated structures, explicit unresolved work, or other conservative signals. Continuations must have finite limits and must be retained in the receipt.

Do not let continuation logic invent new scope merely to keep a model talking.

## Routing

A model may qualify for one category and not another. Task-specific routing is preferred over declaring a model globally best when the evidence only supports narrower conclusions.

## Changes

Keep the harness thin. Reuse external evaluation frameworks where useful, but keep this repository's acceptance policy and evidence format authoritative.

Make new substantive work on branches and expose it through pull requests. When referring to a pull request to the user, include its title with its number; hashes may be included for verification but are not a substitute for the title.

## Output

Terminal output should attempt readable color when attached to a capable terminal. Retained receipts must not depend on ANSI color or other presentation decoration.
