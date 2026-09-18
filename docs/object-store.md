# Local content-addressed object store

`tools/object_store.py` is the first executable storage primitive for Gym's
hosted-service survivability work. The object bytes live in a directory chosen
by the operator; they do **not** live in this public Git repository.

The layout is deterministic:

```text
STORE/
  sha256/
    ab/
      cdef...          # remaining 62 hexadecimal digits
```

The full SHA-256 digest remains the object identity used by model manifests,
run records, and receipts.

## Put an object

```sh
python3 tools/object_store.py put /srv/gym-objects model.gguf
```

The command copies the source into a temporary file while hashing it, flushes
the temporary file, and publishes it at its digest path without overwriting an
already-present object. Re-importing identical bytes reports `PRESENT`.
If an object already occupying the digest path fails hash or size verification,
the import fails rather than replacing it.

A successful `put` prints the digest, byte size, and retained path. Those
values can be copied into the corresponding manifest or receipt.

## Verify an object

```sh
python3 tools/object_store.py verify \
    /srv/gym-objects \
    0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

When the manifest also records the exact size:

```sh
python3 tools/object_store.py verify \
    /srv/gym-objects \
    0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --size-bytes 123456789
```

Verification returns success only when the retained bytes hash to the requested
digest and, when supplied, have the requested size. Missing objects, corrupt
objects, and size mismatches remain explicit failures.

## Boundary

This tool supplies local immutable-object storage and verification only. It does
not decide what may legally be retained or redistributed, encrypt private
evidence, mirror repositories, export GitHub metadata, retain CI artifacts, or
perform the network-isolated recovery drill. Those remain separate
survivability work.

The store itself must still be backed up appropriately. Content addressing
detects changed or missing retained bytes; it is not redundancy.
