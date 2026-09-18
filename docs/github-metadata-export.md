# GitHub metadata snapshots

`tools/export_github_metadata.py` exports explicitly selected GitHub metadata
that matters to Gym provenance or recovery.

It is a GitHub-specific adapter around the `gh` command. The retained output is
plain JSON plus a provider-neutral hash manifest.

The exporter intentionally does **not** archive every object in an account.

## Selection file

Each non-comment row is:

```text
KIND<TAB>OWNER/REPO<TAB>NUMBER
```

Supported kinds are:

- `issue` — issue metadata and issue comments;
- `pull` — pull-request metadata, issue comments, reviews, review comments,
  commits, and changed-file relationships;
- `release` — release metadata by numeric release ID;
- `workflow_run` — run metadata, job metadata, and artifact metadata by run ID.

Example:

```text
issue	fuego-ironworks/gym	2
pull	fuego-ironworks/gym	5
workflow_run	fuego-ironworks/gym	35353268706
```

The numeric identifier is deliberate. It avoids treating a mutable title or tag
string as object identity.

## Export

Authenticate `gh` outside the repository, then run:

```sh
python3 tools/export_github_metadata.py \
    /srv/gym/github-export.tsv \
    /srv/gym/github-snapshots/2026-09-18
```

The output directory must not already exist. The exporter first builds a
process-specific partial directory. Only after every selected endpoint succeeds
does that directory become the requested snapshot.

If any endpoint fails, the partial snapshot is removed and the command fails.

Each repository represented in the selection also gets
`repository.json` containing repository-level metadata.

## Snapshot manifest

The root `snapshot.json` records:

- schema version `gym.github_export.v1`;
- each requested kind/repository/number tuple;
- each retained JSON file;
- its source API endpoint;
- byte size;
- SHA-256 digest.

Paginated GitHub responses are flattened into a single JSON array before being
written, so the retained file does not depend on GitHub's original page size.

## Retention boundary

This exporter captures metadata only.

For workflow runs, `artifacts.json` records artifact metadata but not artifact
ZIP bytes. It also does not retain workflow/job log bytes. Those expire on
GitHub and require a separate retention path into local storage.

For releases, `release.json` contains release and release-asset metadata but
does not download the asset bytes.

Those byte-retention steps should be handled separately so a metadata snapshot
cannot be mistaken for possession of the underlying artifact.

## Privacy and credentials

Do not put `gh` credentials or private exported snapshots in this public
repository.

The export specification may identify private repository objects; store such a
spec with the private recovery set when its contents are sensitive.

The exporter never writes authentication configuration into the snapshot. GitHub
API responses may themselves contain user names, comments, email-like fields,
or other sensitive content, so snapshot storage should follow the sensitivity
of the selected evidence.

## Recovery use

A retained snapshot is useful without GitHub because its JSON and hashes are
ordinary local files.

It does not recreate GitHub as a service. Its purpose is narrower: preserve the
metadata needed to understand acceptance cases, provenance, and historical
evidence after the hosted interface is unavailable.
