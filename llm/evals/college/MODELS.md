# Initial local model matrix

The goal is not to identify a single "best" local model. Local models provide stable, repeatable anchors while hosted models test the systems people actually use.

## Start with two unrelated small model families

### Qwen3 4B

Suggested Ollama tag:

`qwen3:4b-instruct`

Official references:

- https://ollama.com/library/qwen3
- https://huggingface.co/Qwen/Qwen3-4B

The upstream Qwen3-4B model has about 4B parameters. Qwen publishes it under Apache-2.0. Ollama provides 4B local variants.

### Phi-4 mini

Suggested Ollama tag:

`phi4-mini:3.8b`

Official references:

- https://ollama.com/library/phi4-mini
- https://huggingface.co/microsoft/Phi-4-mini-instruct

Phi-4-mini-instruct has about 3.8B parameters and is published by Microsoft under the MIT license. Ollama's default 3.8B Q4_K_M build is about 2.5 GB.

## Why two models

A before/after effect that appears in one family may be a quirk of that model's post-training. Running the same literal prompt pair against unrelated model families makes that visible.

Do not treat agreement between two small models as proof that the effect generalizes to hosted frontier systems.

## Model identity is part of the result

Record more than the friendly model name when possible:

- exact tag;
- local model digest/hash;
- parameter size;
- quantization;
- prompt template;
- local runtime version.

A tag can later point to different bytes. Old result directories should never be silently regenerated under the same run id.

## Thinking versus instruct variants

Do not mix a thinking/reasoning variant with a non-thinking/instruct variant inside what is supposed to be one repeated run. They are different experimental conditions.

## Suggested bootstrap sizes

For development, use a small question subset and 2-3 pairs per question.

For a first stored corpus, 10 repeated pairs per question is enough to expose obvious instability without pretending to be a definitive statistical sample.

Once the harness is stable, add at least one larger local model. Small-model behavior is useful but should not define the project.
