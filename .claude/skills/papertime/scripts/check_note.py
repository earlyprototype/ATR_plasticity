#!/usr/bin/env python3
"""Check a reading note against the house format.

Usage:
    python3 check_note.py NOTE.md [--register REGISTER.md] [--strict]
    python3 check_note.py --self-test

Errors (exit status 1):
    an em dash anywhere in the file;
    no level-one title as the first line;
    no italic standfirst paragraph directly under the title;
    no provenance blockquote (a line beginning "> **Provenance.**") above
        the first section;
    no section whose heading contains "in brief";
    no closing section whose heading contains "what remains";
    no "Sources" section;
    with --register, any hypothesis number (H-number, pattern
        \\bH\\d+[a-z]?\\b) or experiment identifier (EXP-identifier, pattern
        \\bEXP_\\d{3}[a-z0-9]*(-[A-Za-z0-9]+)*\\b) that the register does not
        mention. The patterns are the ones the ATR_research CI uses.

Warnings (exit status 0 unless --strict):
    en dashes; arrows ("->" or the arrow character) outside code; exclamation
    marks outside code; a file name that does not follow
    <TOPIC>_NOTE_<YYYY-MM-DD>.md; sections out of the expected order; a
    paragraph in the "in brief" section that does not open with a bold
    lead-in; a closing section missing one of its four questions; a body with
    no claim marked established, inferred or speculation; a provenance block
    that does not state the marking convention.

The checker is mechanical. It cannot tell whether a number carries its scale
and baseline or whether a term is defined in its sentence; reread the note
for those.
"""

import argparse
import os
import re
import sys

HYP_RE = re.compile(r"\bH\d+[a-z]?\b")
EXP_RE = re.compile(r"\bEXP_\d{3}[a-z0-9]*(?:-[A-Za-z0-9]+)*\b")
FILENAME_RE = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*_NOTE_\d{4}-\d{2}-\d{2}\.md$")
H1_RE = re.compile(r"^#\s+\S")
H2_RE = re.compile(r"^##\s+(.*\S)\s*$")
STANDFIRST_RE = re.compile(r"^(\*(?!\*).+(?<!\*)\*|_(?!_).+(?<!_)_)\s*$")
PROVENANCE_RE = re.compile(r"^>\s*\*\*Provenance\.?\*\*")
EM_DASH = "\u2014"
HORIZONTAL_BAR = "\u2015"
EN_DASH = "\u2013"
ARROW_RE = re.compile("(?:->|\u2192|=>)")


class Report:
    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []

    def error(self, msg, line=None):
        self.errors.append((line, msg))

    def warn(self, msg, line=None):
        self.warnings.append((line, msg))

    def render(self):
        out = []
        for kind, items in (("error", self.errors), ("warning", self.warnings)):
            for line, msg in items:
                where = f"{self.path}:{line}" if line else self.path
                out.append(f"{where}: {kind}: {msg}")
        return out


def strip_code(text):
    """Remove fenced blocks and inline code spans so prose checks do not
    fire on code."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


def split_sections(lines):
    """Return (head_lines, [(heading_text, start_line_no, body_lines), ...]).
    Line numbers are 1-based. The head is everything before the first
    level-two heading."""
    head = []
    sections = []
    current = None
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if line.startswith("```"):
            in_fence = not in_fence
        m = H2_RE.match(line) if not in_fence else None
        if m:
            current = (m.group(1), i, [])
            sections.append(current)
        elif current is None:
            head.append(line)
        else:
            current[2].append(line)
    return head, sections


def paragraphs(lines):
    """Blank-line separated paragraphs as (first_line_no, text)."""
    out = []
    buf = []
    start = None
    for i, line in enumerate(lines, start=1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            out.append((start, "\n".join(buf)))
            buf = []
    if buf:
        out.append((start, "\n".join(buf)))
    return out


def check_text(path, text, register_text=None):
    rep = Report(path)
    lines = text.split("\n")

    # Dashes and characters, everywhere including code.
    for i, line in enumerate(lines, start=1):
        if EM_DASH in line or HORIZONTAL_BAR in line:
            rep.error("em dash; use a comma, a colon, or a new sentence", i)
        if EN_DASH in line:
            rep.warn('en dash; write ranges as "5 to 10" and joins with a hyphen', i)

    # Prose-only characters.
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        prose = re.sub(r"`[^`]*`", "", line)
        if ARROW_RE.search(prose):
            rep.warn("arrow chain; write the relation as a sentence", i)
        if re.search(r"!(?![\[=])", prose):
            rep.warn("exclamation mark; the number beside its baseline does the striking", i)

    # File name.
    base = os.path.basename(path)
    if path != "<string>" and not FILENAME_RE.match(base):
        rep.warn("file name does not follow <TOPIC>_NOTE_<YYYY-MM-DD>.md")

    # Title.
    first = next((l for l in lines if l.strip()), "")
    if not H1_RE.match(first):
        rep.error("the first line must be a level-one heading (# Title) naming the subject", 1)

    head, sections = split_sections(lines)

    # Standfirst: the first paragraph after the title.
    head_paras = paragraphs(head)
    stand = None
    for ln, para in head_paras:
        if H1_RE.match(para.split("\n")[0]):
            continue
        stand = (ln, para)
        break
    if stand is None or not STANDFIRST_RE.match(stand[1].replace("\n", " ")):
        rep.error("no italic standfirst paragraph directly under the title "
                  "(one paragraph wrapped in single asterisks saying what was "
                  "asked, when, for whom, and where the note sits)",
                  stand[0] if stand else None)

    # Provenance block.
    prov_lines = [(i, l) for i, l in enumerate(head, start=1) if PROVENANCE_RE.match(l)]
    if not prov_lines:
        rep.error('no provenance blockquote above the first section '
                  '(a line beginning "> **Provenance.**")')
    else:
        i0 = prov_lines[0][0]
        block = []
        for l in head[i0 - 1:]:
            if l.startswith(">"):
                block.append(l.lstrip("> ").strip())
            else:
                break
        prov_text = " ".join(block).lower()
        for word in ("established", "inferred", "speculation"):
            if word not in prov_text:
                rep.warn(f'the provenance block does not state the marking convention (missing "{word}")', i0)
                break

    # Sections.
    def find(pred):
        for idx, (title, ln, body) in enumerate(sections):
            if pred(title.lower()):
                return idx
        return None

    if not sections:
        rep.error("no level-two sections (## heading)")
    brief = find(lambda t: "in brief" in t)
    closing = find(lambda t: "what remains" in t)
    sources = find(lambda t: re.search(r"\bsources?\b", t) is not None)
    if brief is None:
        rep.error('no section whose heading contains "in brief"')
    if closing is None:
        rep.error('no closing section whose heading contains "what remains"')
    if sources is None:
        rep.error('no "Sources" section')

    if brief is not None and brief != 0:
        rep.warn('the "in brief" section should be the first section', sections[brief][1])
    if sources is not None and sources != len(sections) - 1:
        rep.warn('"Sources" should be the last section', sections[sources][1])
    if closing is not None and sources is not None and closing != sources - 1:
        rep.warn('the closing section should sit directly before "Sources"', sections[closing][1])

    if brief is not None:
        title, ln, body = sections[brief]
        for pln, para in paragraphs(body):
            if para.lstrip().startswith(("**", "__")):
                continue
            if para.lstrip().startswith(("- ", "* ", "1. ")):
                continue
            rep.warn('a paragraph in the "in brief" section does not open with a bold '
                     'lead-in stating the answer', ln + pln - 1 + 0)
    if closing is not None:
        title, ln, body = sections[closing]
        joined = (title + "\n" + "\n".join(body)).lower()
        for phrase in ("what happened", "what it means", "what remains", "decision"):
            if phrase not in joined:
                rep.warn(f'the closing section does not answer "{phrase}"', ln)

    # Epistemic marks in the body (everything after the head).
    body_text = strip_code("\n".join(l for _, _, b in sections for l in b)).lower()
    n_est = len(re.findall(r"\bestablished\b", body_text))
    n_inf = len(re.findall(r"\binferred\b|\ban inference\b", body_text))
    n_spec = len(re.findall(r"\bspeculation\b", body_text))
    if n_est == 0:
        rep.warn("no claim in the body is marked established")
    if n_inf == 0:
        rep.warn("no claim in the body is marked inferred")
    rep.marks = (n_est, n_inf, n_spec)

    # Register.
    if register_text is not None:
        registered = {t.upper() for t in HYP_RE.findall(register_text)} | \
                     {t.upper() for t in EXP_RE.findall(register_text)}
        for i, line in enumerate(lines, start=1):
            for tok in set(HYP_RE.findall(line)) | set(EXP_RE.findall(line)):
                if tok.upper() not in registered:
                    rep.error(f"identifier '{tok}' has no row in the register; a note "
                              f"proposes, the register allocates", i)
    return rep


def self_test():
    good = """# A subject

*A reading note written 2026-09-05 for the operator, in answer to one question.*

> **Provenance.** Read from the record. Nothing here was run. Each claim is marked as established, inferred, or speculation.

---

## 1. The answers in brief

**Yes.** Because of the record. Section 2 has the details.

## 2. The question

The fact is established. The reading is inferred, not measured.

## 3. What remains, and what needs the operator's decision

What happened: a note. What it means: little. What remains: nothing. What needs the operator's decision: none.

## Sources

- The record.
"""
    rep = check_text("GOOD_NOTE_2026-09-05.md", good)
    assert not rep.errors, rep.render()
    assert not rep.warnings, rep.render()
    assert rep.marks == (1, 1, 0), rep.marks

    bad = good.replace("Because of the record", "Because " + EM_DASH + " of the record")
    bad = bad.replace("## Sources", "## Bibliography")
    bad = bad.replace("*A reading note", "A reading note").replace("one question.*", "one question.")
    rep = check_text("bad.md", bad)
    msgs = " | ".join(m for _, m in rep.errors)
    assert "em dash" in msgs and "Sources" in msgs and "standfirst" in msgs, msgs
    assert any("file name" in m for _, m in rep.warnings)

    reg = "| H1 | something |\n| EXP_001 | something |\n"
    noted = good.replace("The fact is established.", "H1 and EXP_001 are established; H2 is not registered.")
    rep = check_text("GOOD_NOTE_2026-09-05.md", noted, reg)
    msgs = [m for _, m in rep.errors]
    assert len(msgs) == 1 and "'H2'" in msgs[0], msgs

    warn = good.replace("**Yes.** Because", "Yes. Because").replace("marked as established, inferred, or speculation", "marked")
    rep = check_text("GOOD_NOTE_2026-09-05.md", warn)
    kinds = " | ".join(m for _, m in rep.warnings)
    assert "bold" in kinds and "marking convention" in kinds, kinds
    print("self-test OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a reading note against the house format.")
    ap.add_argument("note", nargs="?", help="the markdown note to check")
    ap.add_argument("--register", help="identifier register; every H-number and "
                    "EXP-identifier in the note must appear in it")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--self-test", action="store_true", help="run the checker's own tests")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.note:
        ap.error("a note path is required (or --self-test)")
    text = open(args.note, encoding="utf-8").read()
    register_text = open(args.register, encoding="utf-8").read() if args.register else None
    rep = check_text(args.note, text, register_text)
    for line in rep.render():
        print(line)
    est, inf, spec = rep.marks
    print(f"{args.note}: {len(rep.errors)} errors, {len(rep.warnings)} warnings; "
          f"marks: established {est}, inferred {inf}, speculation {spec}")
    failed = bool(rep.errors) or (args.strict and bool(rep.warnings))
    print("FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
