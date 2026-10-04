# Final-prompt-token neighborhood scan

Only the output of the named component at the **last prompt token** is zeroed;
all other token positions and components retain their original outputs. The
original three BME continuation pairs are scored for both the canonical and
held-out BME student at step512 and step1000. Three short neutral continuations
measure collateral language-model loss under the identical intervention.

All MLP layers 0–7 plus 12 and 23 are tested; attention layers 0–5 plus 8 and
23 provide neighboring and spaced comparisons. This is a bounded component
scan, not a claim that untested components are irrelevant. The layer-0
step1000 canonical score is checked against the independent token-scope run.
`scan.tsv` and `pairs.tsv` retain aggregate and candidate-level numbers.
Zeroing a component is causal for this likelihood score; it does not identify
a unique semantic representation or a training origin.
