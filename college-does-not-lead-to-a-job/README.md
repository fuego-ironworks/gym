# College Does Not Lead to a Job

This branch audits a narrow question:

> When a college program markets itself as preparation for a career, what employment obligation does the institution actually undertake?

The title is a hypothesis/prompt, not a universal empirical claim. Distinguish
**career marketing, school-reported counts, independently inspected career
outcomes, use of purchased training and an employment promise or contract**.
Each answers a different question.

## For every program, record separately

1. program and credential;
2. tuition/net price if available;
3. career language used in marketing;
4. career services promised;
5. internships/projects required;
6. whether internship placement is guaranteed;
7. whether full-time employment is guaranteed;
8. whether tuition/refund depends on employment;
9. whether any employer is contractually committed to hire;
10. reported placement outcomes;
11. denominator and reporting window;
12. students excluded from the denominator;
13. date the pages were checked.

Never turn "no guarantee found on the pages reviewed" into "the university has legally promised nothing anywhere." Preserve the scope of the search.

## Why this matters

Career-oriented language can be strong without creating a job contract.

Examples include:

- "accelerate your career";
- "prepares you for entry into";
- "find your place";
- "unrivaled placement";
- employer-logo lists;
- median salary claims;
- testimonials.

Keep those as attributed institutional claims until their underlying career
events are inspected. They do not establish either training value or:

> If you complete this program, we or a named employer will employ you.

Pythia should learn to distinguish those statements automatically.

## Seed case 1: Cornell Systems Engineering M.Eng. — Mechatronic & Robotics Systems pathway

Cornell's Systems Engineering M.Eng. lets students design a "Mechatronic & Robotics Systems" pathway.

Cornell Engineering markets the professional M.Eng. broadly with "Accelerate your career" and says it fosters connections among students, alumni, and potential employers. The Systems Engineering admissions material says the M.Eng. is self-funded.

Official pages reviewed:
- https://www.engineering.cornell.edu/sys/meng/meng-pathways/
- https://www.engineering.cornell.edu/meng
- https://www.engineering.cornell.edu/sys/se/programs/meng-degree-distance-learning/admission
- https://www.engineering.cornell.edu/sys/se/alumni/get-involved

### Current finding

No explicit contract or guarantee of a post-graduation job was found on the official program/career pages reviewed.

This answers the contract question within the reviewed source scope. Employer
connections, actual outcomes and training value require separate evidence.

Status: `no_job_guarantee_found_in_reviewed_materials`.

## Seed case 2: UC Berkeley Haas Master of Financial Engineering

Haas uses unusually strong career language. Current pages describe "unparalleled job placement," tell students to "find your place in a leading finance or technology firm," and advertise dedicated career services and employer relationships.

Official pages:
- https://mfe.haas.berkeley.edu/
- https://mfe.haas.berkeley.edu/careers
- https://mfe.haas.berkeley.edu/careers/mfe-placements
- https://mfe.haas.berkeley.edu/admissions/faq

### Current finding

The FAQ explicitly states that internships are **not guaranteed**, even though an internship or approved on-site project is required for graduation.

For the Class of 2025, Haas reports:

- 70 students looking for jobs;
- 67 received offers;
- 67 accepted offers.

These are school-reported offer events among job seekers. They do not establish
verified starts, retention, course use, satisfaction or a good bargain. The three
outside the acceptance count remain unexplained. A positive-sounding aggregate
must not erase a contrary account of actual work.

No explicit guarantee of full-time post-graduation employment was found on the official pages reviewed.

Status: `no_full_time_job_guarantee_found_in_reviewed_materials`.

## Training signal

When a user hears:

> "This program has 96% placement."

Pythia should ask:

- 96% of whom?
- by what date?
- offer, acceptance, actual start or retained job?
- did the actual duties use the purchased curriculum?
- did the graduate want the job and consider the transaction worthwhile?
- did the job require unwanted relocation or further credentials?
- what jobs?
- what salary distribution?
- how many did not receive one?
- is placement a reported historical outcome, a forecast, or a contractual promise?
- what exactly does the school owe a student who graduates without the advertised outcome?

## Corrected inference case

[Berkeley placement and the Paleologo account](cases/berkeley-placement-inference-regression.md)
records the failure of using offer counts to reassure about the work or price.
The criteria are ready for a source-grounded evaluation; no model replay or
passing model receipt has been produced by this documentation change.

## Cross-links

- `trick-majors`
- `regret`
- `quants`
- `finance`
- `law-school`
