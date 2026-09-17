# Durable record schemas

This directory defines the first provider-neutral records used by Gym's survivability work:

- `model-manifest.schema.json` identifies an exact model revision and the hashes of locally retained model objects without storing those objects in Git;
- `run.schema.json` binds a work item to an exact model manifest, runner, runtime, scorers, and policies before execution;
- `receipt.schema.json` records the observed result, including `PASS`, `FAIL`, or `UNKNOWN`, and hashes any retained outputs or traces.

A `sha256` field is the lowercase SHA-256 digest of the exact retained bytes it names. Hashing a JSON record means hashing that file's exact bytes; this first slice does not define a semantic JSON canonicalization rule.

Private trace payloads and large model objects are deliberately not represented as inline content. A receipt may retain only their hashes and mark trace references `private`; the payload belongs in a local/private evidence store. Model weights belong in a content-addressed local object store subject to their license, not in this public repository.

The schemas use a deliberately small subset of JSON Schema Draft 2020-12 so they can be checked with the standard-library-only validator in `tools/validate_record.py`. The validator fails closed if a schema begins using an unsupported keyword.

Run all validation tests locally with:

```sh
python3 -m unittest discover -s tests
```

Validate one record with:

```sh
python3 tools/validate_record.py schemas/run.schema.json path/to/run.json
```
