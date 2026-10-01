# Layer-0 token-position recheck

The earlier exploratory layer-0 claim has no recorded hook or token-position
definition. This run tests three distinct intervention scopes on the exact
Pythia-410M step512/step1000 three-pair likelihood contrast:

- `prompt`: every prompt token's layer-0 MLP output;
- `last_prompt`: the last prompt token's layer-0 MLP output;
- `continuation`: only the teacher-forced continuation positions.

`zero` deletes the chosen output region. `replace` copies the same region from
the other checkpoint, run on the identical token sequence. The recipient's
other positions remain intact. `pairs.tsv` records all per-candidate scores,
pair and aggregate margins, neutral score changes for ablation, and an exact
token-boundary check. The self-replacement control must leave the baseline
unchanged under all three scopes.

The previous all-position scan remains a separate receipt. None of these
whole-vector substitutions identifies a unique feature or a training cause.
If one scope reverses the score, that observation is specific to the stated
tokens, continuations, checkpoint direction, and model output hook.
