# Vast.ai public API

Checked: 2026-09-29.

Vast.ai exposes a public API for renting and controlling GPU compute. For `gym`, it can serve as a replaceable hosted-compute provider for training, qualification, and other GPU jobs. The provider API is infrastructure evidence only; Vast model listings, benchmarks, and marketing claims do not establish model capability.

## Current public surfaces

Vast currently exposes three related interfaces:

1. **REST API** — raw GPU marketplace, account, instance, volume, and related operations.
2. **Official CLI / Python SDK** — the `vastai` package and command wrap the REST API.
3. **Serverless inference** — OpenAI-compatible inference endpoints, separate from renting a raw GPU instance.

For automation in this repository, prefer the official SDK or CLI unless raw HTTP is materially useful. Vast currently mixes API versions across operations, so a wrapper gives some protection against endpoint drift.

Official sources:

- REST/API documentation: <https://docs.vast.ai/>
- OpenAPI source: <https://github.com/vast-ai/docs/tree/main/api-reference/openapi/yaml>
- Official CLI and SDK: <https://github.com/vast-ai/vast-cli>
- API-key management: <https://console.vast.ai/manage-keys/>
- Billing FAQ: <https://console.vast.ai/faq/>

## Authentication

Create an API key in the Vast console. Current REST definitions primarily use an HTTP bearer token:

```http
Authorization: Bearer $VAST_API_KEY
```

Some older endpoint definitions and CLI internals still accept an `api_key` query parameter. New code should prefer the `Authorization` header and should never place an API key in Git, a retained receipt, command output committed to the repository, or a public CI log.

Since October 2025, newly created Vast API keys are viewable only once. Vast also supports fine-grained API-key permissions. Retain only a key identifier when provenance requires one, not the secret itself.

The Python SDK can read `VAST_API_KEY` or an explicitly supplied key. The CLI can store a key with:

```sh
vastai set api-key "$VAST_API_KEY"
```

## Versioning caveat

Vast does not currently present one uniform REST version.

The current official OpenAPI definitions include, among others:

| Operation | Method and path |
| --- | --- |
| Search GPU offers | `POST /api/v0/bundles/` |
| Create an instance from an offer | `PUT /api/v0/asks/{offer_id}/` |
| List instances | `GET /api/v1/instances/` |
| Show one instance | `GET /api/v0/instances/{instance_id}/` |
| Start or stop an instance | `PUT /api/v0/instances/{instance_id}/` |
| Destroy an instance | `DELETE /api/v0/instances/{instance_id}/` |
| Show current user | `GET /api/v0/users/current/` |

The OpenAPI files use `https://console.vast.ai` as the production server. Other current Vast material also shows `cloud.vast.ai` and versioned examples. Treat the raw HTTP host/path pair as versioned provider state, not as a permanent interface.

The instance-list endpoint is already on v1 and uses keyset pagination. It returns at most 25 instances per page and supplies `next_token` for the next page. Code must consume all pages when it needs a complete account view.

## Offer search

Offer search is the main pricing and machine-discovery call:

```
POST https://console.vast.ai/api/v0/bundles/
```

Useful filters for `gym` include:

- `gpu_name`
- `gpu_ram`
- `num_gpus`
- `dph_total`
- `reliability`
- `verification`
- `rentable`
- `duration`
- `cuda_max_good`
- `geolocation`
- `static_ip`
- `direct_port_count`
- `cpu_cores`
- `cpu_ram`
- `disk_space`
- `inet_up` and `inet_down`
- `storage_cost`
- `inet_up_cost` and `inet_down_cost`

The API accepts comparison objects such as `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, and `notin`.

Example:

```sh
curl -sS \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  https://console.vast.ai/api/v0/bundles/ \
  -d '{
    "limit": 3,
    "verified": {"eq": true},
    "rentable": {"eq": true},
    "num_gpus": {"eq": 1}
  }'
```

For a retained job receipt, record the complete selected offer or at least the offer ID plus the price, GPU, memory, location, reliability/verification, storage price, bandwidth prices, and maximum available duration. A later offer search cannot reconstruct the exact market state that existed when the job launched.

## Create an instance

Creating an instance accepts an offer:

```
PUT https://console.vast.ai/api/v0/asks/{offer_id}/
```

The request can specify a Docker image, disk size, label, launch mode, initial state, environment/port mappings, start command, and related options.

Minimal example:

```sh
curl -sS \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -X PUT \
  "https://console.vast.ai/api/v0/asks/$OFFER_ID/" \
  -d '{
    "image": "pytorch/pytorch:latest",
    "disk": 40,
    "runtype": "ssh",
    "target_state": "running",
    "label": "gym"
  }'
```

A successful response includes `new_contract`, the new instance ID.

Offer availability can change between search and creation. Treat an unavailable offer as a normal marketplace race: re-run the constrained search rather than weakening constraints silently.

## Observe and control an instance

Show one instance:

```sh
curl -sS \
  -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/"
```

List all instances with v1 pagination:

```sh
curl -sS \
  -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v1/instances/?limit=25"
```

Starting and stopping both use:

```
PUT /api/v0/instances/{instance_id}/
```

with a JSON body whose `state` is `running` or `stopped`.

Destroying uses:

```
DELETE /api/v0/instances/{instance_id}/
```

Destruction is the cleanup operation for an ephemeral job. Stopping an instance stops active GPU rental charges, but Vast continues charging storage while the instance exists. A stopped instance therefore must not count as successful cleanup for short-lived `gym` jobs.

## CLI

Current official installation:

```sh
curl -fsSL https://vast.ai/install.sh | bash
```

or:

```sh
python -m pip install vastai
```

Useful lifecycle commands:

```sh
vastai show user
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true'
vastai create instance "$OFFER_ID" --image pytorch/pytorch:latest --disk 40 --ssh --direct
vastai show instance "$INSTANCE_ID"
vastai show instances --raw
vastai stop instance "$INSTANCE_ID"
vastai start instance "$INSTANCE_ID"
vastai destroy instance "$INSTANCE_ID" -y
```

Use `--raw` for machine-readable output. Human table output should never become the parsing contract for a `gym` adapter.

## Python SDK

The current CLI and SDK share the `vastai` package:

```python
from vastai import VastAI

vast = VastAI()

offers = vast.search_offers(
    query="num_gpus=1 reliability>0.99 verified=true rentable=true"
)

instances = vast.show_instances()
```

The SDK also exposes instance creation and lifecycle calls such as `create_instance`, `show_instance`, `start_instance`, `stop_instance`, and `destroy_instance`.

For new repository code, isolate Vast behind a narrow provider adapter rather than leaking SDK objects into qualification logic.

## Serverless API

Vast also exposes serverless inference and an official `Serverless` client. Vast announced OpenAI-compatible serverless endpoints in April 2026.

This interface answers a different problem from raw GPU rental:

- **GPU rental** gives `gym` a machine on which it controls the model, runtime, checkpoints, and artifacts.
- **Serverless** gives `gym` an inference endpoint managed by Vast.

Do not treat a serverless model name as equivalent to a locally controlled model artifact unless the endpoint provides enough immutable revision/provenance information for the intended evidence boundary.

## Billing semantics

Vast separates:

1. active GPU rental cost;
2. storage cost;
3. bandwidth cost.

GPU rental is billed while the instance is active. Storage is billed for every second the instance exists, including while stopped. Bandwidth is billed by transferred bytes. A zero account balance can stop instances without destroying them, which means storage charges can continue.

A provider adapter should therefore distinguish at least:

- `running`
- `stopped`
- `destroyed`
- `cleanup_failed`

and should make destruction part of the normal success and failure cleanup path.

## Rental type

Vast supports on-demand and interruptible/bid-style rentals.

Interruptible instances can stop when outbid or displaced. A `gym` job may use them only when the job is restartable and retained checkpoints/results survive interruption. The receipt should record the rental type and any interruption.

## Recommended `gym` receipt fields

A Vast-backed run should retain enough provider-neutral and provider-specific evidence to explain cost and reproduce the machine choice:

- provider: `vast`
- client surface: REST, CLI, or SDK
- client version
- REST endpoint version(s) used
- offer search filters and ordering
- selected offer ID
- instance/contract ID
- rental type
- GPU model, count, and VRAM
- CUDA/driver capability reported by the offer
- CPU/RAM/disk allocation
- verification and reliability fields
- location
- quoted hourly price
- storage and bandwidth prices
- Docker image or immutable image reference
- creation time
- running time
- terminal state
- destroy attempt/result
- interruption or scheduling events
- API errors and retry count
- cost observed after completion when available

Never retain the API-key secret.

## Minimal adapter boundary

A first `gym` provider adapter only needs a small interface:

```text
search_offers(requirements) -> offers
create_instance(offer, job) -> instance
show_instance(instance) -> state
destroy_instance(instance) -> result
```

Add start/stop, copying, volumes, webhooks, serverless, or billing calls only when a concrete job needs them.

For a one-shot training job, the important control flow is:

```text
search constrained offers
    -> select offer
    -> create instance
    -> wait for running
    -> run workload
    -> copy/retain outputs and receipt
    -> destroy instance
    -> verify destroyed
```

The cleanup verification matters because `stopped` still incurs storage cost.

## Current-change notes

Recent Vast changes relevant to an adapter:

- **October 2025:** API keys became one-time viewable; API responses stopped returning secret key values.
- **April 2026:** Vast announced OpenAI-compatible serverless endpoints and warned that instance listing would move to paginated responses.
- **May 2026:** Vast merged the Python SDK and CLI into the `vastai` package/repository.
- **August 2026:** Vast added programmatic notification and webhook management for lifecycle and account events.

Raw REST integration should pin observed behavior with tests because this surface is changing. The CLI/SDK source and the official OpenAPI repository provide the best current references when prose documentation disagrees.


## Grease client

Grease now has a matching thin REST client on
`dilapidated-shed/grease:vast` in `commands/vast/vast.ysh` (PR #47,
**Hook Grease into the Vast.ai REST API**). Its acceptance workflow runs the
exact pinned Grease/YSH executable against an HTTP fixture and checks the
request method, path, bearer header, JSON body, and missing-key failure.

That client intentionally remains transport-only. `gym` should own job
selection, receipt policy, retry/cleanup policy, and model qualification rather
than pushing those decisions into the shell command.
