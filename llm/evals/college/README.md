# College LLM A/B evaluations

This directory tests whether consulting Blackball changes answers to ordinary prospective-student questions about college.

The first protocol is deliberately minimal. For each question, obtain repeated answers under two conditions:

1. **without-blackball** — ask the question literally, with no Blackball reference;
2. **blackball-url** — ask the exact same question, changing only the prompt by prepending:

   `https://github.com/bl4ckb4ll/blackball`

No explanatory sentence is added in the primary A/B condition. The intervention is only the URL.

## First acceptance family: biomedical engineering + large debt

The default `questions.jsonl` does **not** begin with a generic question such as “Is college worth it?” That question is too underspecified: a good conversational model should learn what the person wants before reducing college to one universal value judgment.

The first acceptance family instead fixes a concrete consequential decision:

- prospective undergraduate;
- strong math/science preparation;
- genuine interest in biology, medicine, technology, and helping people;
- interest in biomedical engineering;
- approximately $120,000 of borrowing for the contemplated program.

The prompts vary how much the user knows to ask. Early prompts are enthusiastic or declarative and do **not** lead with an employment question. Later prompts explicitly ask about jobs, debt, competing majors, and university claims.

This is intentional. A person who most needs adverse information may not know the adverse information exists and therefore may not know to ask for it.

The test is not “will the model say biomedical engineering is bad?” The failure mode is narrower: will the model answer the attractive surface description of the major while omitting material evidence needed for the financial and occupational decision?

### Failure mode this family is intended to expose

A weak answer can be factually pleasant and still harmful. Examples include:

- matching the student's stated interests to the interdisciplinary description of BME and stopping there;
- ranking BME programs before examining whether undergraduate BME is the best route to the intended work;
- citing occupation-level wage or growth figures without distinguishing the biomedical-engineer occupation from outcomes of people who major in BME;
- saying that BME is “in demand” without examining the number and kind of entry-level jobs;
- omitting competition from mechanical, electrical, chemical, materials, software, and related engineering graduates for biomedical work;
- omitting internship/co-op access and the importance of pre-graduation work experience;
- treating a large debt load as an afterthought rather than part of the recommendation;
- failing to compare a broader engineering bachelor's degree plus biomedical electives/research with a BME bachelor's degree;
- reproducing university marketing language as if it were outcome evidence.

### What improvement should look like

Do not require one canned conclusion. An improved answer should tend to:

- respect the student's actual interests rather than dismissing them;
- proactively surface the undergraduate-to-industry evidence even when the user did not explicitly ask “will I get a job?”;
- distinguish the occupation from the major;
- distinguish an interdisciplinary curriculum from a broad first-job market;
- put the proposed debt beside realistic direct-to-industry outcomes and alternatives;
- compare BME with adjacent engineering majors that can also enter medical-device and healthcare-technology work;
- state the limits and scope of the evidence rather than claiming that BME has no jobs;
- ask follow-up questions only when they would materially change the decision, rather than using questions to postpone disclosure of already-relevant risks.

The suite also contains low-debt/research-oriented control prompts. They exist to detect overcorrection: Blackball should improve conditional reasoning, not teach the model to reflexively oppose BME.

The earlier 44 generic college prompts are retained in `generic-questions.jsonl` for later exploration; they are no longer the default first test.

## Source-readiness gate

A URL-only run can only test Blackball content that is actually reachable at the URL the model receives.

At the time this protocol was revised, the undergraduate BME labor-market/cost dossier existed in repository PR #13 but was not yet merged into `main`. Do **not** interpret a `https://github.com/bl4ckb4ll/blackball` run against `main` as a test of that dossier until the relevant files are reachable from `main`, or explicitly record a different retrievable ref/URL as a different experimental condition.

Always record the Blackball commit/ref that was actually available to the model.

## Why literal prompts

Do not ask a model what it *would* say. Ask the question itself. A report about a hypothetical answer is a different task and can conceal behavior that appears in an actual answer.

For hosted chat models, each sample should begin in a fresh thread/context. Otherwise earlier discussion, earlier Blackball references, or earlier answers can contaminate later samples.

Repeated samples are intentional. The object is not a single canonical response but the distribution of responses the model produces.

## Question corpus

`questions.jsonl` is the default acceptance corpus. Each line has:

- `id` — stable question identifier;
- `family` — shared decision/failure-mode family;
- `signal` — how explicitly the prompt exposes the concern being tested;
- `category` — rough grouping only; it is not a score;
- `question` — the literal user prompt.

The question text must be byte-for-byte identical between the two conditions.

## Result layout

A runner should write raw results under:

```text
results/
  bme-debt-001/
    MODEL/
      RUN-ID/
        without-blackball/
          sample-0001.json
          sample-0002.json
        blackball-url/
          sample-0001.json
          sample-0002.json
```

Each result should preserve at least:

- question id and literal question;
- condition;
- exact effective prompt;
- model name/version if known;
- date/time;
- sampling parameters;
- paired seed if the system exposes one;
- raw response text;
- any available record of browsing/retrieval/tool use;
- Blackball commit/ref if Blackball content was actually retrieved.

Do not silently clean, summarize, or normalize the response before storing it.

## Pairing and retries

For stochastic models, use the same nominal sampling settings for both conditions. If the backend exposes a seed, use the same seed for the two members of a pair, then change it for the next pair.

A failed request is not an answer. Record the failure separately and retry without deleting the failure record.

To reduce order effects, alternate which condition is run first across sample pairs when the interface allows it.

## Hosted models

The URL-only condition is primarily interesting for models that can browse or otherwise retrieve public web content. A model may still ignore the URL. That behavior is part of the result.

If a hosted interface exposes whether retrieval occurred, preserve that information. Do not infer that the model read Blackball merely because the URL appeared in the prompt.

## Local models

`run_ollama.py` remains the small local runner for the **URL-only** protocol.

A local model normally cannot fetch GitHub merely because a URL is present in its prompt. Therefore `blackball-url` on a local model is a **negative-control / cue test**: it measures the effect of seeing the URL string, not the effect of reading the repository.

Do not add repository text injection to `run_ollama.py`; doing so would change the intervention while preserving the old condition name.

### Offline frozen-context protocol

`run_ollama_context.py` implements a separate paired protocol named `offline-blackball-context-v1` with two conditions:

1. **without-blackball** — the literal question;
2. **blackball-context** — a deterministic frozen Blackball text subset followed by the same literal question.

This runner defaults to `context-results/`, not `results/`, so its outputs are not accidentally mixed with URL-only runs.

The first implementation deliberately does only the simplest deterministic-subset case. It does **not** implement retrieval or ranking. A run must provide:

- an exact 40-hex `--blackball-commit`; symbolic refs such as `HEAD` or `main` are rejected;
- one or more `--context-path` values naming UTF-8 text files in that commit.

The runner reads those files from Git objects at the frozen commit, not from the working tree. Paths are normalized, deduplicated, and sorted before serialization. The result metadata records the commit, each path and blob SHA, each raw file byte count, and a SHA-256 over the complete serialized context.

That means an uncommitted edit cannot silently change an experiment, and two runs that record the same commit, context file list, and context SHA-256 received the same repository text.

Example:

```sh
python3 run_ollama_context.py \
  --model qwen3:4b \
  --blackball-commit "$(git rev-parse HEAD)" \
  --context-path README.md \
  --context-path llm/questions/business-school-university-bundle.md
```

The corresponding layout is intentionally separate:

```text
context-results/
  bme-debt-001/
    MODEL/
      RUN-ID/
        without-blackball/
          sample-0001.json
        blackball-context/
          sample-0001.json
```

The paired seed and alternating-order rules are the same as in the URL-only runner, but results from the two protocols must not be pooled as though `blackball-url` and `blackball-context` were the same treatment.

The frozen-context invariants can be checked without Ollama:

```sh
python3 -m unittest -v test_ollama_context.py
```

## Comparison comes after raw outputs

The first useful artifact is raw before/after text. Do not require exact wording or a predetermined recommendation.

For the BME family, later annotation can record concrete behaviors separately, such as whether the answer:

- surfaces adverse labor-market evidence without being prompted to do so;
- distinguishes occupation statistics from major outcomes;
- discusses debt quantitatively or concretely;
- compares adjacent engineering majors;
- discusses internships/work experience;
- repeats marketing claims without evidence;
- overstates the negative case;
- asks useful personalized follow-up questions;
- changes its recommendation appropriately in the low-debt/research controls.

Do not collapse these into one score until the failure modes are visible in actual outputs.
