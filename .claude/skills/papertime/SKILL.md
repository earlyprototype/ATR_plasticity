---
name: papertime
description: Papertime. Write an operator-facing reading note (the paper format) that answers research questions from the record and the literature in the ATR house format, then check it and build its shareable page. Use whenever the user asks a research question that deserves a written answer rather than a chat reply, asks for a note, briefing, primer, write-up, reading note or explainer on a paper, method, model or tool, asks "how different is X from Y", "could our harness run on Z", "does this technique apply to our model", "where does this sit in the literature", or asks to turn an answer you already gave into a document or a page. Triggers on "papertime", "/papertime", "paper time", "paper format", "reading note". Also use when asked to check an existing note against the format, to rebuild a note's page, or when a repository's rules say operator-facing answers land as reading notes. Reach for it even when the user does not say "note": a multi-part research question with a knowledgeable but non-specialist reader is this skill's case.
---

# Papertime

Papertime writes a reading note. A reading note is the written form of a research answer for one reader: the
operator of a project, sharp and attentive, without a machine-learning
background, and the final authority on what happens next. The note exists so
that person can decide something. Everything in the format serves that: the
answer comes first, every claim says how it is known, every number carries its
scale and a baseline, and the note ends by saying what remains and what only
the operator can decide.

The markdown file is the record. The page built from it is a convenience for
reading and sharing; where they differ, the file governs.

## Before writing

1. Read `references/voice.md`. Those eight rules are not style preferences;
   they are what makes the note usable by its reader. Two of them are the ones
   most often broken: define every term and identifier in the sentence that
   uses it, and never write an em dash anywhere.
2. Read `references/format.md` for the shape of the note and the reasons for
   each part. `assets/TEMPLATE_NOTE.md` is a skeleton to start from.
3. Research before drafting, and keep a provenance trail as you go: which
   file, which page, which command produced each fact. The provenance block
   at the top of the note is written from that trail, and "nothing here was
   run" is a sentence you can only write if it is true. Verify what can be
   verified: read the model configuration rather than recalling it, check the
   file listing rather than assuming the file exists, quote the paper's
   wording rather than paraphrasing from memory.
4. If the repository has an identifier register (for example
   `_STAGE2_JSPACE/REGISTER.md` with hypothesis numbers like H16 and
   experiment identifiers like EXP_011), use only identifiers that already
   have a row there. Never coin one inside a note; a note proposes, the
   register allocates. The checker can verify this for you.

## Writing

Follow `references/format.md` section by section. In short: a title that
names the subject; an italic standfirst saying what was asked, for whom and
where the note sits; a provenance block; "the answers in brief", one bold
lead-in per question asked; one numbered section per question with tables
where numbers compare; claims marked inline as established, inferred or
speculation; and a closing section that answers, in this order, what
happened, what it means, what remains and what needs the operator's decision.
Then sources.

Write the whole note before polishing any part of it. Answers first, then
evidence, then limits. Use the project's founding analogy only where it
carries the idea, and say where it stops holding.

## Checking

Run the checker on the finished file:

```bash
python3 scripts/check_note.py docs/MY_NOTE_2026-09-05.md [--register path/to/REGISTER.md] [--strict]
```

Paths here are relative to this skill's directory: installed as a plugin the
script is `${CLAUDE_PLUGIN_ROOT}/skills/papertime/scripts/check_note.py`,
and vendored into a repository it is
`.claude/skills/papertime/scripts/check_note.py`. `--self-test` runs the
checker's own tests.

It fails on em dashes, missing structural parts (title, standfirst,
provenance block, the answers-in-brief section, the closing section, sources)
and, when a register is given, on any hypothesis or experiment identifier
that has no register row. It warns on missing epistemic marks and on en
dashes. Fix what it reports and run it again; a clean run is the bar for
committing. The checker is mechanical and cannot judge the writing; reread
the note once as the reader would, and find the sentence a smart outsider
would stumble on.

## Building the page

The page builder turns the markdown into a designed, theme-aware HTML page
with the epistemic marks rendered as small tags, a sticky section list and a
scrolling table container:

```bash
python3 -m pip install markdown   # once
python3 scripts/build_note_page.py docs/MY_NOTE_2026-09-05.md --out /tmp/my_note.html \
    --title "Short Name" --for "TC, the operator" --project "ATR project" --preview
```

Give `--title` a short, specific name (two to four words) for the browser tab
and gallery; the page's heading still carries the note's full title. `--for`
and `--project` fill the line above the title and may be left out. Relative
links to other markdown files are rewritten to the repository's GitHub URLs
when the note lives in a git checkout with an `origin` remote; pass
`--repo-url` and `--branch` to override. `--preview` also writes a standalone
`.preview.html` you can open in a browser to check the page before publishing.
To embed a figure, put an HTML fragment beside the note and list it in a
sidecar `<note>.figures.json` (see `references/format.md`); the builder picks
the sidecar up by name.

Publish the built file with the Artifact tool (pass a one-emoji favicon and a
one-sentence description; the artifact starts private) and give the user
the link beside the file path. If the environment has no Artifact tool, the
built file itself is the deliverable.

## Landing it in a repository

- File name: `docs/<TOPIC>_NOTE_<YYYY-MM-DD>.md`, uppercase topic, the date it
  was written.
- Add one pointer line to the document readers would start from (a primer,
  an index, a README section) so the note is discoverable.
- Commit the note in its own commit; the page is not committed unless the
  repository keeps pages.
- The PR body states what the note answers, what it does not change (usually
  no code or results), any register or closing-line conventions the
  repository requires, and the decisions it puts to the operator.
- Post nothing about the note's conclusions anywhere else until the operator
  has read it; a note is advice to one reader first.

## What not to do

- Do not restate the source paper or the repository's own findings at length;
  point to them and say what is new.
- Do not pad with hedges; mark the claim's status once, inline, and move on.
- Do not put numbers in prose without their scale and baseline, and do not
  put them in prose at all when a table would carry them.
- Do not end with an offer or a summary of the summary; end with the four
  closing questions and stop.
