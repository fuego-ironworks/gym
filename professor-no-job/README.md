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
- `k12_education`
- `research_institute_nonacademic`
- `government_nonacademic`
- `private_sector`
- `nonprofit_nonacademic`
- `military`
- `trade_or_service`
- `self_employed`
- `other_nonacademic`
- `unknown`

"Real job" is too ambiguous for the data file. Preserve at least two questions separately:

1. Did the person have documented paid employment **outside postsecondary academia**?
2. Did the person have documented paid employment **outside education/research institutions altogether**?

That distinction matters for cases such as K-12 teachers and private research institutes.

## Domain-practice test

For faculty and administrators attached to explicitly occupational programs, ask a third question:

> Has this person actually worked in the occupation or industry the program is preparing students to enter?

Examples include:

- quantitative finance / banking / trading / risk;
- human resources;
- emergency services;
- nursing and allied health;
- engineering specialties;
- skilled trades;
- teaching;
- management and business programs.

Record separately:

- the occupational domain being taught or administered;
- documented employment in that domain outside education;
- consulting/advisory work;
- research about the domain;
- licenses/certifications where relevant;
- duration and recency of practice.

Do not silently equate scholarship about an industry with employment in that industry, and do not silently equate absence of industry employment with inability to teach. The purpose is to expose the distinction and later compare it with curriculum, student outcomes, cost, and the claims programs make to prospective students.

The person-level version of this test lives in `cases.csv`; detailed employment rows live in `people.csv`.

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

A published CV that lists only academic employment supports the narrower statement **"no nonacademic employment is listed on this CV."** It does not prove that no such employment ever occurred.

## Derived questions

Once enough cases exist, compute:

- fraction with any documented nonacademic paid work;
- fraction with work outside education/research institutions altogether;
- fraction of occupational-program leaders with documented practice in the occupation;
- years of domain practice before teaching/administering the program;
- years of nonacademic work before first faculty appointment;
- fraction moving directly from school/postdoc into academia;
- differences by field;
- differences by birth cohort;
- differences between faculty and university presidents;
- common kinds of nonacademic work among the exceptions.

Keep the raw rows visible so the aggregate claim can be checked against the underlying evidence.
