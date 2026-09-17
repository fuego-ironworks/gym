# Hosted-service survivability

## Goal

Gym should remain useful if a hosted service becomes unavailable, expensive, capacity-constrained, legally unavailable, or simply no longer fits the workflow.

This is a design assumption, not a prediction about any particular company or service.

GitHub, GitHub Actions, model-hosting sites, proprietary model APIs, and other hosted infrastructure may accelerate the system. They must not be the only place where acceptance-critical state exists.

The durable target is a local model workbench whose corpus, provenance, routing evidence, retained models, and recovery procedure survive the loss of any single hosted provider.

## Failure model

The recovery design should separately consider loss of:

1. the GitHub web service;
2. GitHub-hosted Actions runners;
3. GitHub-specific metadata such as issues, pull requests, releases, workflow logs, and artifacts;
4. a model download host or registry;
5. a proprietary inference API;
6. a particular local inference runtime;
7. one local disk or machine.

Do not collapse these into one vague "backup" requirement. Different state is lost in each case.

A normal Git clone protects Git objects. It does not by itself preserve issue discussion, pull-request metadata, Actions history, workflow artifacts, releases, or other hosted state.

## Recovery set

The local recovery set should be sufficient to reconstruct the useful system without contacting the original providers.

### Repository state

For every repository that materially participates in qualification or provenance, retain a complete Git mirror including branches and tags.

The mirror is the authoritative recovery copy of Git data. An ordinary working checkout is disposable.

Where non-Git GitHub state matters to evidence or reconstruction, export it separately in a documented, machine-readable form. Candidate material includes:

- issues and issue comments used as acceptance cases or provenance;
- pull-request metadata, discussions, reviews, and changed-file relationships;
- release metadata and retained release assets that matter to a test;
- workflow-run metadata;
- acceptance-critical workflow logs and artifacts.

Do not indiscriminately archive every hosted object forever merely because it exists. Retention should be driven by reproducibility, current acceptance claims, regression value, and recovery value.

### Model state

A model manifest should identify the exact runnable object rather than relying on a human-readable model name.

A useful manifest includes, where applicable:

```text
model family / public name
upstream source
upstream revision or commit
download or acquisition date
weight-file object hashes
tokenizer/config object hashes
quantization / format
runtime requirements
license / terms metadata
known local conversion steps
conversion-tool revisions
resulting converted-object hashes
hardware qualification notes
```

Large weights belong in a content-addressed local object store, not in this public Git repository.

The manifest belongs in Git when it contains no private or restricted material. Exact cryptographic hashes make retained objects verifiable without trusting filenames or the continued existence of the original model host.

Model licenses and distribution restrictions remain authoritative. Local retention for recovery does not create permission to publish or redistribute weights.

### Evaluation and workbench state

Provider-neutral retained state should include enough information to reconstruct a model run and understand why it affected routing.

Depending on the task, that can include:

- work-item identifier;
- prompt and relevant context identity;
- attached-file or repository revisions;
- model manifest identity;
- inference runtime and runtime revision;
- generation settings;
- raw model outputs;
- automatic continuations;
- tool calls and returned evidence needed by the evaluation;
- deterministic checks;
- critic / verifier outputs;
- synthesis or selection steps;
- scorer versions and results;
- routing decision before and after the run;
- explicit user corrections;
- resulting commits, pull requests, tests, merges, reverts, or repairs where known.

Do not infer missing provenance. Missing links stay unknown.

Private conversation history, credentials, and other sensitive traces do not belong in the public repository. Gym should define schemas and import/export procedures without publishing private evidence.

## Local runner boundary

Every acceptance-critical workflow that normally executes on hosted CI should have a local execution path.

This does not require duplicating every convenience job. It does require avoiding a state where a model is considered qualified only because an opaque hosted runner once produced a green badge that can no longer be reproduced.

Prefer runner definitions that can execute from a normal local shell with explicit dependencies and produce plain receipts.

Hosted CI may invoke the same runner. The hosted wrapper should remain thinner than the durable local path.

## Receipts

A retained receipt should remain intelligible without the original web interface.

Where practical, bind the receipt to exact inputs with hashes:

- repository revision;
- case definition;
- model manifest;
- retained model objects;
- runtime / adapter revision;
- scorer and policy revisions;
- produced artifact hashes.

Terminal color is presentation only. Durable receipts must remain plain-text or machine-readable without ANSI decoration.

A hosted URL may be retained as provenance, but a URL alone is not a receipt when the underlying evidence can expire.

## Automation

The steady state should require little routine human maintenance.

A scheduled local maintenance pass may:

1. update complete Git mirrors;
2. export newly relevant hosted metadata;
3. retain acceptance-critical CI logs and artifacts before they expire;
4. inspect model manifests for missing retained objects;
5. verify content-addressed objects by hash;
6. ingest newly observed model failures, corrections, and successful work into candidate qualification cases;
7. run conservative model requalification;
8. run a small recovery self-test.

Failure to mirror or export should be visible as a failed maintenance receipt. It must not silently delete old evidence or silently strengthen an acceptance claim.

## Recovery drill

The system should periodically test the failure assumption rather than merely documenting it.

A representative drill should deliberately deny access to:

- GitHub;
- GitHub Actions;
- the original model host;
- proprietary inference APIs.

Starting only from the retained local recovery set, the drill should be able to:

1. restore Gym and the repositories/cases needed by the fixture;
2. verify retained objects by hash;
3. locate at least one exact locally runnable model revision;
4. execute a representative acceptance case locally;
5. emit new durable receipts;
6. reconstruct which model, input, runner, checks, scorer, and routing policy produced the result.

The drill passes only for what it actually exercises. One successful fixture does not prove that every repository, model, or historical receipt is recoverable.

## Rebuild hierarchy

Classify state by how it is recovered:

### Irreplaceable or expensive-to-reconstruct

Retain redundantly.

Examples include user corrections, provenance links, accepted/rejected work history, private evaluation cases, legally retained model objects that may disappear upstream, and acceptance-critical receipts.

### Rebuildable but costly

Retain when storage is cheaper than reconstruction, while keeping deterministic rebuild instructions.

Examples include converted model formats, indexes, embeddings, compiled runners, and derived corpora.

### Cheaply rebuildable

Prefer recipes and hashes over redundant copies.

Examples include ordinary working trees, caches, temporary build directories, and generated summaries whose inputs remain intact.

## Provider replacement

Provider-neutral schemas should make replacement boring.

An adapter may know how to talk to a particular model API, GitHub, model registry, or local inference runtime. The rest of Gym should reason in terms of work items, model manifests, retained objects, checks, receipts, and routing evidence.

Replacing one provider should not require rewriting the evaluation corpus or throwing away historical receipts.

## Acceptance rule

Hosted independence is a property to test, not a slogan.

Gym should eventually have a recovery acceptance case that runs without network access and proves a narrow end-to-end path from retained repository state and retained model objects through local inference, checking, scoring, and receipt generation.

Until such a drill exists, documentation establishes the intended boundary but does not establish that recovery actually works.
