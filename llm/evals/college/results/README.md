# Raw evaluation results

Store generated answers here without editing them.

Directory convention:

```text
QUESTION-ID/MODEL/RUN-ID/without-blackball/sample-NNNN.json
QUESTION-ID/MODEL/RUN-ID/blackball-url/sample-NNNN.json
```

The two files with the same sample number form a pair and should use the same nominal sampling settings and seed when the backend exposes one.

Hosted-model results captured manually should follow the same structure where practical. Preserve the exact prompt and raw answer in each record. If the interface does not expose model version, seed, system prompt, browsing state, or other metadata, record those fields as unknown rather than guessing.

Do not overwrite old runs. Model behavior, hidden system instructions, retrieval behavior, and the Blackball repository can all change with time.
