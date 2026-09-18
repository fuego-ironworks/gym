# Local Git mirrors

`tools/mirror_git.py` maintains complete bare mirrors for repositories whose Git
history participates in Gym qualification or provenance.

The tool is provider-neutral. Each manifest row is:

```text
RELATIVE_PATH<TAB>REMOTE
```

Blank lines and lines beginning with `#` are ignored.

Example:

```text
fuego-ironworks/gym.git	https://github.com/fuego-ironworks/gym.git
fuego-ironworks/agent-tamagotchi.git	https://github.com/fuego-ironworks/agent-tamagotchi.git
```

Do not put credentials in a manifest committed to Git. A private local manifest
may use whatever authenticated Git transport the operator has configured.

## Update mirrors

```sh
python3 tools/mirror_git.py update \
    /srv/gym/git-mirrors \
    /srv/gym/mirrors.tsv
```

A missing destination is created with `git clone --mirror`. An existing mirror
is updated with pruning.

After each clone or update the tool requires:

- a bare Git repository;
- the exact origin recorded in the manifest;
- the mirror refspec `+refs/*:refs/*`;
- a successful `git fsck --full`.

Each successful row prints a plain receipt containing the relative mirror path,
number of retained refs, and a SHA-256 digest of the sorted ref-name/object-id
listing.

The tool deliberately refuses to silently retarget an existing mirror when the
manifest URL changes. A remote migration should be an explicit operation rather
than a side effect of routine backup maintenance.

## Offline verification

```sh
python3 tools/mirror_git.py verify \
    /srv/gym/git-mirrors \
    /srv/gym/mirrors.tsv
```

`verify` performs the local structure, origin/refspec, object-integrity, and ref
receipt checks without contacting the remote. This is the recovery-side check:
it remains usable when the original host is unavailable.

## Scheduling

The update command is intentionally an ordinary local command so the durable
path does not depend on GitHub Actions or any particular scheduler.

For a machine using cron, a local installation can run, for example:

```cron
17 3 * * * cd /opt/gym && python3 tools/mirror_git.py update /srv/gym/git-mirrors /srv/gym/mirrors.tsv >> /srv/gym/receipts/git-mirror.log 2>&1
```

Choose paths and cadence for the actual machine. The repository does not claim a
scheduled backup exists until a local scheduler has actually been configured and
its receipt retained.

## Evidence boundary

A complete Git mirror preserves Git refs and reachable Git objects. It still
does not preserve GitHub issues, pull-request discussions/reviews, workflow logs,
artifacts, releases, or other hosted metadata.

Those remain separate work under the survivability issue. A green mirror receipt
must not be promoted into a claim that the whole hosted project state is
recoverable.
