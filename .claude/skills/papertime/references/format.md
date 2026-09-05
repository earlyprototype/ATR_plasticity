# The reading-note format, section by section

Every part below exists for the reader named in `voice.md`: one operator who
will decide something after reading. Keep the order; each part answers a
question the reader has at that moment.

## 1. Title

A heading that names the subject, not the occasion: "Latent context, small
chat models, and the J-space on GPT-2", not "Notes from Friday's questions".

## 2. Standfirst

One italic paragraph directly under the title. It says what was asked, when,
for whom, and where the note sits among the project's other documents ("It
sits beside JSPACE_PRIMER.md, which explains the paper itself, and does not
repeat it"). This is where the reader learns whether to read on.

## 3. Provenance block

A blockquote beginning `**Provenance.**` that says where every class of fact
came from and how it was checked: which files in the record, which paper
pages or web pages, which configuration files, which listings, on which day.
State plainly whether anything was run. End it with the marking convention:
each claim is marked as established (read from a record or a paper), inferred
(reasoned from established facts), or speculation.

## 4. The answers in brief

A section whose heading contains "in brief". One paragraph per question the
operator asked, each opening with a bold lead-in that states the answer as a
sentence, followed by one to three sentences of the essential why, and a
pointer to the section that carries the detail. A reader who stops here has
the answers; everything after is evidence and limits.

## 5. One section per question

Numbered sections, in the order the questions were asked. Inside each:

- Lead with the answer again, then the evidence, then the limits.
- Define every term and identifier in the sentence that uses it, every time.
- Give every number its scale and a baseline: "0.013 span share against a
  0.252 chance level", never "0.013".
- Put comparisons in tables; put wide tables in the note as markdown tables
  and let the page scroll them.
- Mark claims inline: "This is established.", "That is an inference, not a
  measurement.", "Speculation, flagged by the paper itself:". The page
  builder renders these phrases as tags, so use the exact words established,
  inferred (or "an inference"), and speculation.
- Where the project has a founding analogy, use it only where it carries the
  idea, and say where it stops holding.
- Correct the operator's premise where it is wrong, plainly and early
  ("Your rugby memory is two experiments fused"), then answer the question
  they meant.

## 6. What remains, and what needs the operator's decision

The closing section answers four questions in this order and then stops:

1. What happened (what the note did, what was found, including new facts).
2. What it means (the reading, with its status marked).
3. What remains (numbered, ordered by information per unit cost, each with a
   cost estimate, none of it started).
4. What needs the operator's decision (only decisions that are genuinely
   theirs; each with the choice stated and, where you have one, your
   recommendation marked as offered).

No closing offer, no summary of the summary.

## 7. Sources

A bulleted list of everything cited, with URLs where they exist and file
paths for the project's own record.

## Mechanical conventions

- No em dashes anywhere. Use a comma, a colon, or a new sentence. Number
  ranges use "to" ("5 to 10"), not a dash.
- No bare identifiers or acronyms: "EXP_011, the J-space overlap experiment".
- Percentages carry their counts: "64 percent, 27 of 42".
- Dates are ISO (2026-09-05) in file names and provenance; prose may spell
  them out.
- Relative links to other markdown files in the same repository are allowed;
  the page builder rewrites them to GitHub URLs.

## Figures

The page builder accepts an optional sidecar `<note>.figures.json` beside the
note:

```json
[
  { "after": "Second, in depth the workspace band", "file": "figures/depth_band.html" }
]
```

`after` is the opening words of the paragraph the figure follows, as plain
text before any formatting; `file` is an HTML fragment (a `<figure>` element
with a `<figcaption>`, inline SVG welcome) relative to the sidecar, or give
`html` with the fragment inline. Draw figures to one scale and label every
value. Take colours from the page's CSS variables so the figure reads in both
themes: `--fig-1` (blue), `--fig-2` (grey) and `--fig-3` (gold) for data,
`--ink-2` for labels, `--surface` for gaps between cells; the text classes
`svg-lbl` and `svg-num` and the line classes `svg-tick` and `svg-cell` are
styled by the page. Give every shape a plain fill as well, as a fallback. A
figure earns its place only when it shows a mechanism the prose cannot; the
worked example's depth-band figure is `docs/figures/depth_band.html` beside
the note.

## Pull request body pattern

```
## What this adds
<one paragraph: the note, its date, the questions it answers>

## What this does not change
No code, no results, no experiment artifacts. Nothing was run.

<any repository closing-line convention, e.g. "No-Close: a reading note, not tied to an issue.">
```

## Worked example

`docs/LATENT_CONTEXT_NOTE_2026-09-04.md` in
`earlyprototype/lucier-gpt2-activ-tensor-reson-experiments`, with its page
built by this skill's builder. Read its first two screens to calibrate the
register and the density before writing your own.
