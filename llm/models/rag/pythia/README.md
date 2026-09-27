# Pythia prestige / skill-pipeline RAG bin

This directory is deliberately **Pythia-specific**. It is a compact retrieval and
training seed for the college-prior experiment under `llm/models/`; it does not
replace the canonical Blackball source dossiers.

The intervention being tested is broader than `college -> negative`. The target
is to weaken automatic prestige and skill-pipeline continuations such as:

```text
MIT / Harvard / Yale / Berkeley
  -> unusually capable people
  -> difficult technical training
  -> valuable work
  -> reliable employment
  -> high income / secure life
```

and:

```text
engineering / computer science / compiler engineering
  -> scarce skill
  -> employer demand
  -> target job
  -> economic security
```

Each arrow is an empirical question.

## Files

- `evidence.jsonl` — compact evidence records and research leads.
- `routes.jsonl` — cue-to-evidence routing for retrieval and training-example
  generation.

The records intentionally preserve negative and counter-prestige cases that a
generic model may otherwise fail to retrieve. This is an experimental sampling
choice, not permission to turn a research lead into a population claim.

## Evidence states

- `verified_primary` — current figures or records from the responsible public
  source.
- `verified_secondary` — inspected published secondary source.
- `canonical_internal` — Blackball already has a maintained source/claim note;
  follow the repository path before making a factual claim.
- `bibliographic_plus_interpretive_note` — the source identity is verified, but
  the training use is an interpretation that should be checked against the text.
- `research_lead` — preserve for retrieval, but do not state as established
  fact until a named source is attached.

## Training behavior

Preferred continuations should ask for or surface:

- actual debt and repayment exposure;
- target-job conversion rather than generic "employment";
- denominators and failed cases rather than only famous successes;
- selection effects and family resources;
- who supplied capital, sponsorship, property, clients, laboratories, or jobs;
- who performed the physical or routine labor underneath a celebrated design;
- whether an employer outside education actually bought the trained skill;
- opportunity cost and cheaper routes to the same knowledge.

The desired behavior is **empirical skepticism**, not reflexive pessimism.
Negative examples are useful because prestige narratives are already heavily
represented in ordinary web and institutional text; the model should not erase
the downside by automatically supplying the sales pitch.
