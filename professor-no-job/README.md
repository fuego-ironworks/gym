# Professor No Job

A collection of academic CVs and biographies used to test a simple question:

> Before becoming a professor or university administrator, what paid work did this person actually do outside academia?

The branch name is intentionally provocative. The data should not be.

## Rule

Do not code the conclusion in advance.

For every person, find the best available CV, institutional biography, personal site, archived résumé, or other reliable employment record and classify every documented paid role.

A professor with substantial nonacademic work is a counterexample and belongs in the collection just as much as someone whose entire adult career is inside universities.

## Population

Collect examples encountered elsewhere in Gym work, plus deliberate samples of:

- professors;
- deans;
- provosts;
- university presidents;
- institute directors;
- prominent public-facing academics.

Do not restrict the sample to cases expected to support the hypothesis.

## Employment classes

Use these categories:

- `academic_faculty`
- `academic_staff`
- `academic_administration`
- `student_or_trainee`
- `government_nonacademic`
- `private_sector`
- `nonprofit_nonacademic`
- `military`
- `trade_or_service`
- `self_employed`
- `other_nonacademic`
- `unknown`

"Real job" is too ambiguous for the data file. The operational question is **documented nonacademic paid employment**.

## Evidence standard

For each claim record:

- person;
- current/most notable academic role;
- institution;
- source URL;
- source type;
- years if known;
- employer;
- job title;
- employment class;
- whether the role was paid, if documented;
- notes;
- uncertainty.

Never infer that a person has never held a nonacademic job merely because a short university biography omits one. Prefer a full CV. If the evidence is incomplete, mark it incomplete.

Likewise, do not assert that someone has "never done a job interview." That generally is not recoverable from a CV unless the person has explicitly discussed it.

## Derived questions

Once enough cases exist, compute:

- fraction with any documented nonacademic paid work;
- years of nonacademic work before first faculty appointment;
- fraction moving directly from school/postdoc into academia;
- differences by field;
- differences by birth cohort;
- differences between faculty and university presidents;
- common kinds of nonacademic work among the exceptions.

Keep the raw rows visible so the aggregate claim can be checked against the underlying evidence.
