# Clearer item definitions in prompt.txt

## Intent

Haiku and Opus disagree on about 1 in 10 item codes, mostly because the prompt leaves boundaries open
(applicant traits vs job features, colleagues vs customers, fixed vs rotating shifts). Rewrite the
definitions so a careful coder reaches the same decision every time.

What the codes are for:
- **GoldinTraits**: test Goldin's theory at firm level. Does the job likely need the worker at
  particular times, at short notice, with others depending on them? We assume what the firm
  advertises is how the job is.
- **IrregularHours, flexibility**: measure what firms advertise, to be checked later against a survey.
- In all cases: code what the post says.

## Approaches

- **A. Rewrite each definition in place with "Counts" / "Does not count" lines (chosen).** Fixes the
  unclear boundaries at a few hundred extra input tokens per post.
- B. A plus 2–3 fully coded example posts. Best for consistency, but roughly doubles input cost and
  risks copying the examples. Kept as a later step if a pilot still shows rule breaks.
- C. Goldin items in a separate call. Doubles requests; poor fit for a subscription.

## Decisions (from the conversation)

General
- Applicant traits count when they describe what the job values ("Du er selvstændig").
- Boilerplate still excluded: "gode kolleger", "spændende opgaver", text about the company's other
  roles. "Fleksibel arbejdsplads" is no longer excluded (it counts for flexibilityHoursEmployee).
- A busy company ("har travlt") counts for the job.

IrregularHours
- Evening: occasional evenings count ("aftenmøder", "enkelte aftener").
- Night: "sovevagt" and "døgnvagt" count.
- ShiftWork: anything called vagt or hold, fixed or rotating, day-only included ("dagvagt",
  "daghold", "aftenhold"). On-call duty and "skiftende arbejdstider" are not ShiftWork.

flexibility
- flexibilityHoursCompany: overtime, on-call duty (rådighedsvagt, tilkaldevagt), "skiftende
  arbejdstider", "efter behov", and "Du er fleksibel" when tied to working hours.
- flexibilityHoursEmployee: worker chooses or plans own hours, including general flexibility promises
  ("fleksibel arbejdsplads", "rige muligheder for fleksibilitet"). Shift work, including influence on
  one's own shift plan, is not flexibility. Part-time is not flexibility.
- flexibleTasks: **dropped**.

GoldinTraits (Goldin's definitions, plus why each matters)
- timePressure: strict deadlines; "travlt" implies deadlines. Targets alone do not count.
- contactWithOthers: only when the job requires contact to be done. Generic teamwork or
  "godt samarbejde" does not count.
- interpersonalRelationships: the same people over time who rely on the worker (own customers,
  residents followed over time, parents). Colleagues are not mentioned (revised after review).
- contactWithOthers, colleagues: count only when working with them is the task itself
  ("tværfagligt samarbejde om borgeren").

Added after the prompt review
- present = 1 only from the post's own words; never infer from occupation or industry.
- Hours flexibility is decided by who controls the hours (required → Company, offered → Employee).
- Company-wide benefits apply to the job; contact details, "hold" meaning team, negated mentions,
  and "administration af medicin" do not count. English posts use the same items.
- Accepted extra cues: timePressure "højt tempo", "arbejde under pres", "mange bolde i luften",
  "travle perioder"; flexibilityHoursCompany "travle perioder", "træde til med kort varsel",
  "tilkaldevikar".
- Administrative narrowed: clerical paperwork for compliance (forms, registering, reporting), not
  professional law or accounting roles. "bogføring" and "kontorarbejde" removed as cues; both now
  listed under "does not count".
- structuredWork: the job sets tasks, priorities and goals for the worker. "Du er struktureret"
  does not count.
- freedomToMakeDecisions: the worker decides without supervision; traits like "selvstændig" count.

OtherTraits
- Administrative: complying with regulation, forms, case processing. Care documentation
  ("dagbogsnotater") does not count.
- AI: unchanged.
- **partTime (new)**: counts for "deltid", hours under 37 stated, "fuldtid eller deltid", student
  jobs, "fleksjob". Does not count: temp or fixed-term without hours, hourly pay or
  "tilkaldevikar" without hours.

## Changes

- `scripts/llm_topics/prompt.txt`: rewritten as above.
- `scripts/llm_topics/annotate_jobnet.py`: remove `flexibleTasks` from `Flexibility`, add `partTime`
  to `OtherTraits`.
- README in `scripts/llm_topics`: still 15 items; check any mention of flexibleTasks.

## Consequences

- New `PROMPT_ID`, so new pilots go to new folders; existing results are kept.
- Haiku and Opus pilots must be re-run before `compare_pilots.py` can compare them on the new prompt.

## Left to the builder

- Exact wording and choice of Danish examples per item.
