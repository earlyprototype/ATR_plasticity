#!/usr/bin/env python3
"""Build a designed, theme-aware HTML page from a reading note.

Usage:
    python3 build_note_page.py NOTE.md [--out PAGE.html] [--title "Short Name"]
        [--repo-url https://github.com/owner/repo] [--branch main]
        [--figures NOTE.figures.json] [--for "TC, the operator"]
        [--project "ATR project"] [--preview]

The markdown file governs; the page is a view of it. The builder:
  reads the title (first level-one heading), the italic standfirst and the
    provenance blockquote from the head of the note;
  renders the body with python-markdown (tables, fenced code, heading ids);
  rewrites relative links to markdown files into GitHub URLs, using the
    note's git checkout (origin remote and default branch) unless --repo-url
    or --branch say otherwise;
  sets the "in brief" section as a lead block;
  wraps every table in a scrolling container;
  renders the inline epistemic marks (established, inferred or an inference,
    speculation) as small tags;
  inserts figures listed in a sidecar JSON file after the paragraph each one
    names (see --figures);
  writes the page body without <html>, <head> or <body> tags, ready for the
    Artifact tool, and with --preview also writes a standalone
    <PAGE>.preview.html for a local browser.

Needs the "markdown" package: python3 -m pip install markdown
"""

import argparse
import html
import json
import os
import posixpath
import re
import subprocess
import sys

try:
    import markdown
except ImportError:
    sys.exit("build_note_page.py needs the markdown package: python3 -m pip install markdown")


# ----------------------------------------------------------------------------
# git helpers: where the note lives, so relative links can point at GitHub

def git(cwd, *args):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def repo_context(note_path):
    """Return (repo_url, branch, note_rel_path) or (None, None, None)."""
    d = os.path.dirname(os.path.abspath(note_path))
    top = git(d, "rev-parse", "--show-toplevel")
    if not top:
        return None, None, None
    url = git(d, "remote", "get-url", "origin")
    m = re.match(r"^(?:git@github\.com:|https?://github\.com/)([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    repo_url = f"https://github.com/{m.group(1)}/{m.group(2)}" if m else None
    head = git(d, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    branch = head.split("/", 1)[1] if "/" in head else "main"
    rel = os.path.relpath(os.path.abspath(note_path), top).replace(os.sep, "/")
    return repo_url, branch, rel


# ----------------------------------------------------------------------------
# the note's head: title, standfirst, provenance

def parse_head(text):
    lines = text.split("\n")
    title = ""
    standfirst = ""
    provenance = ""
    body_start = 0
    i = 0
    n = len(lines)
    # title
    while i < n and not lines[i].strip():
        i += 1
    if i < n and lines[i].startswith("# "):
        title = lines[i][2:].strip()
        i += 1
    # head paragraphs until the first level-two heading
    head_end = i
    while head_end < n and not lines[head_end].startswith("## "):
        head_end += 1
    head_lines = lines[i:head_end]
    paras, buf = [], []
    for l in head_lines + [""]:
        if l.strip():
            buf.append(l)
        elif buf:
            paras.append("\n".join(buf))
            buf = []
    for p in paras:
        flat = " ".join(x.strip() for x in p.split("\n"))
        if not standfirst and re.match(r"^\*(?!\*).+\*$", flat):
            standfirst = flat[1:-1]
        elif not provenance and flat.startswith(">"):
            body = " ".join(x.lstrip("> ").strip() for x in p.split("\n") if x.strip() != ">")
            provenance = re.sub(r"^\*\*Provenance\.?\*\*\s*", "", body)
    body_start = head_end
    body_md = "\n".join(lines[body_start:])
    return title, standfirst, provenance, body_md


def inline_html(md_text):
    out = markdown.markdown(md_text)
    return re.sub(r"^<p>|</p>$", "", out.strip())


# ----------------------------------------------------------------------------
# transforms on the rendered body

def rewrite_md_links(html_text, repo_url, branch, note_rel):
    if not repo_url:
        return html_text
    note_dir = posixpath.dirname(note_rel or "")

    def sub(m):
        href = m.group(1)
        if re.match(r"^(?:[a-z]+:|#|/)", href):
            return m.group(0)
        path, _, frag = href.partition("#")
        if not path.endswith(".md"):
            return m.group(0)
        target = posixpath.normpath(posixpath.join(note_dir, path))
        new = f"{repo_url}/blob/{branch}/{target}" + (f"#{frag}" if frag else "")
        return f'href="{new}"'

    return re.sub(r'href="([^"]+)"', sub, html_text)


def mark(cls, word):
    return f'<span class="mark {cls}">{word}</span>'


def mark_claims(body):
    est, inf, spec = "est", "inf", "spec"
    rules = [
        (r"(<p>|<li>)(Established)(?=[,: ])", lambda m: m.group(1) + mark(est, m.group(2))),
        (r"(<p>|<li>)(Inferred)(?=[,: ])", lambda m: m.group(1) + mark(inf, m.group(2))),
        (r"(<p>|<li>)(Speculation)(?=[,: ])", lambda m: m.group(1) + mark(spec, m.group(2))),
        (r"\b(is|are|was|were|remains|facts are|both are|all are) established\b",
         lambda m: m.group(1) + " " + mark(est, "established")),
        (r"\b(is|are|was|were|remains) inferred\b", lambda m: m.group(1) + " " + mark(inf, "inferred")),
        (r"\binferred, not\b", lambda m: mark(inf, "inferred") + ", not"),
        (r"\bis an inference\b", lambda m: "is " + mark(inf, "an inference")),
        (r"\ban inference from\b", lambda m: mark(inf, "an inference") + " from"),
        (r"\b(is|remains|It is|That is|This is) speculation\b",
         lambda m: m.group(1) + " " + mark(spec, "speculation")),
        (r"\((established|inferred|speculation)\)",
         lambda m: "(" + mark({"established": est, "inferred": inf, "speculation": spec}[m.group(1)], m.group(1)) + ")"),
    ]
    for pat, fn in rules:
        body = re.sub(pat, fn, body)
    return body


def wrap_brief(body):
    m = re.search(r'(<h2 id="[^"]*">[^<]*in brief[^<]*</h2>)(.*?)(?=<h2 |\Z)', body, re.S | re.I)
    if not m:
        return body
    return body[:m.start()] + m.group(1) + '<div class="brief">' + m.group(2) + "</div>" + body[m.end():]


def insert_figures(body, figures, base_dir):
    for fig in figures:
        after = fig.get("after", "")
        frag = fig.get("html")
        if frag is None and fig.get("file"):
            frag = open(os.path.join(base_dir, fig["file"]), encoding="utf-8").read()
        if not frag:
            print(f"figure skipped: no html or file for {fig!r}", file=sys.stderr)
            continue
        needle = "<p>" + html.escape(after, quote=False)
        i = body.find(needle)
        if i < 0:
            print(f'figure skipped: no paragraph starting "{after}"', file=sys.stderr)
            continue
        j = body.index("</p>", i) + 4
        body = body[:j] + frag + body[j:]
    return body


def table_of_contents(body):
    items = []
    for hid, text in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body):
        label = re.sub(r"^\d+\.\s*", "", re.sub(r"<[^>]+>", "", text))
        items.append(f'<li><a href="#{hid}">{label}</a></li>')
    return "".join(items)


# ----------------------------------------------------------------------------
# the page

CSS = r"""
:root{
  --bg:#F6F7F9; --surface:#FFFFFF; --ink:#171A21; --ink-2:#525A6B; --rule:#D8DCE4; --rule-2:#E9ECF2;
  --accent:#1F4FD8; --accent-soft:#E6ECFB;
  --est:#146C5B; --est-bg:#E1F1EB; --inf:#8A5300; --inf-bg:#F6ECD9; --spec:#5C43A8; --spec-bg:#ECE7F8;
  --fig-1:#1F4FD8; --fig-2:#CBD3E2; --fig-3:#D9A441;
  --band-ws:var(--fig-1); --band-sens:var(--fig-2); --band-motor:var(--fig-3);
  --code-bg:#EEF1F6;
  --font-display:'Newsreader',Georgia,'Times New Roman',serif;
  --font-body:'Source Sans 3','Segoe UI',Helvetica,Arial,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0F1218; --surface:#161A23; --ink:#E6E9F0; --ink-2:#A4AAB9; --rule:#2B3242; --rule-2:#212735;
    --accent:#8AA6FF; --accent-soft:#1C2745;
    --est:#63D2B4; --est-bg:#12312A; --inf:#E9B45C; --inf-bg:#3A2B10; --spec:#BCA6F8; --spec-bg:#2A2244;
    --fig-1:#5E7FF0; --fig-2:#3A4356; --fig-3:#C99A3F; --code-bg:#1D2230;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#0F1218; --surface:#161A23; --ink:#E6E9F0; --ink-2:#A4AAB9; --rule:#2B3242; --rule-2:#212735;
  --accent:#8AA6FF; --accent-soft:#1C2745;
  --est:#63D2B4; --est-bg:#12312A; --inf:#E9B45C; --inf-bg:#3A2B10; --spec:#BCA6F8; --spec-bg:#2A2244;
  --fig-1:#5E7FF0; --fig-2:#3A4356; --fig-3:#C99A3F; --code-bg:#1D2230;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 var(--font-body);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}
a:hover{text-decoration-thickness:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:2px}
.page{max-width:1120px;margin:0 auto;padding:44px 24px 80px;display:grid;grid-template-columns:200px minmax(0,46rem);column-gap:56px;row-gap:0}
.mast{grid-column:1/-1;border-bottom:1px solid var(--rule);padding-bottom:28px;margin-bottom:36px}
.eyebrow{font:600 12px/1.4 var(--font-body);letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);margin:0 0 14px;display:flex;flex-wrap:wrap;gap:8px 18px}
.eyebrow span+span::before{content:"";display:inline-block;width:4px;height:4px;border-radius:50%;background:var(--ink-2);margin:0 10px 3px 0;vertical-align:middle}
h1{font:600 clamp(30px,4.2vw,44px)/1.12 var(--font-display);letter-spacing:-.01em;margin:0 0 18px;max-width:22ch;text-wrap:balance}
.stand{font:400 clamp(18px,1.6vw,20px)/1.5 var(--font-display);font-style:italic;color:var(--ink-2);max-width:60ch;margin:0}
.stand a{color:inherit}
.prov{margin:24px 0 0;max-width:60ch;font-size:14.5px;line-height:1.55;color:var(--ink-2);border-top:1px solid var(--rule-2);padding-top:14px}
.prov .lbl{display:block;font:600 11px/1 var(--font-body);letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);margin-bottom:8px}
.legend{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin-top:12px;font-size:14px;color:var(--ink-2)}
nav.toc{position:sticky;top:28px;align-self:start;font-size:14px;line-height:1.45}
nav.toc .lbl{font:600 11px/1 var(--font-body);letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);margin:0 0 12px}
nav.toc ol{list-style:none;margin:0;padding:0;counter-reset:sec;display:flex;flex-direction:column;gap:8px}
nav.toc li{counter-increment:sec;display:grid;grid-template-columns:20px 1fr;gap:6px;color:var(--ink-2)}
nav.toc li::before{content:counter(sec);font-variant-numeric:tabular-nums;font-weight:600;color:var(--accent)}
nav.toc a{color:var(--ink);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
.prose>p,.prose>ul,.prose>ol,.prose>h2,.prose>h3,.prose>.brief,.prose>blockquote{max-width:34rem}
.prose h2{font:600 clamp(24px,2.4vw,29px)/1.2 var(--font-display);letter-spacing:-.005em;margin:52px 0 16px;padding-top:20px;border-top:1px solid var(--rule);text-wrap:balance}
.prose h2:first-of-type{margin-top:0;border-top:0;padding-top:0}
.prose h3{font:600 19px/1.3 var(--font-body);margin:34px 0 10px;text-wrap:balance}
.prose p{margin:0 0 16px}
.prose ul,.prose ol{margin:0 0 18px;padding-left:1.4em;display:flex;flex-direction:column;gap:9px}
.prose li>p{margin:0}
.prose strong{font-weight:600}
.prose blockquote{margin:0 0 18px;padding:2px 0 2px 18px;border-left:2px solid var(--rule);color:var(--ink-2)}
.prose hr{border:0;border-top:1px solid var(--rule-2);margin:28px 0}
.brief{display:flex;flex-direction:column;gap:14px;margin-bottom:8px}
.brief p{margin:0;padding-left:18px;border-left:2px solid var(--accent)}
.brief p>strong:first-child{font:600 19px/1.35 var(--font-display);display:inline;margin-right:.3em;color:var(--ink)}
code{font:0.86em var(--font-mono);background:var(--code-bg);padding:.08em .35em;border-radius:3px}
pre{background:var(--code-bg);border:1px solid var(--rule-2);border-radius:4px;padding:14px 16px;overflow-x:auto;margin:0 0 20px;max-width:34rem}
pre code{background:none;padding:0;font-size:13.5px;line-height:1.55}
.mark{display:inline-block;font:600 10.5px/1 var(--font-body);letter-spacing:.07em;text-transform:uppercase;padding:4px 6px 3px;border-radius:3px;vertical-align:baseline;position:relative;top:-1px;white-space:nowrap}
.mark.est{color:var(--est);background:var(--est-bg)}
.mark.inf{color:var(--inf);background:var(--inf-bg)}
.mark.spec{color:var(--spec);background:var(--spec-bg)}
.table-scroll{overflow-x:auto;margin:6px 0 22px;border:1px solid var(--rule);border-radius:4px;background:var(--surface)}
table{border-collapse:collapse;font-size:14px;line-height:1.4;font-variant-numeric:tabular-nums;width:max-content;min-width:100%}
th,td{padding:9px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule-2)}
th{font:600 11.5px/1.3 var(--font-body);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);background:var(--bg);position:sticky;top:0}
tbody tr:last-child td{border-bottom:0}
td:first-child{white-space:nowrap}
.prose figure{margin:8px 0 24px;padding:18px 18px 12px;background:var(--surface);border:1px solid var(--rule);border-radius:4px;max-width:46rem}
.prose figure svg{width:100%;height:auto;display:block}
.svg-lbl{font:12px var(--font-body);fill:var(--ink-2)}
.svg-num{font:11.5px var(--font-mono);fill:var(--ink-2)}
.svg-tick{stroke:var(--ink-2);stroke-width:1}
.svg-cell{stroke:var(--surface);stroke-width:2}
figcaption{font-size:14px;line-height:1.5;color:var(--ink-2);margin-top:10px;max-width:60ch}
.foot{grid-column:2;margin-top:48px;padding-top:16px;border-top:1px solid var(--rule);font-size:13.5px;color:var(--ink-2)}
@media (max-width:860px){
  .page{grid-template-columns:minmax(0,1fr);padding:28px 18px 64px}
  nav.toc{position:static;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid var(--rule-2)}
  .foot{grid-column:1}
}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto!important}}
html{scroll-behavior:smooth}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400'
         '&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400&display=swap">')


def build(args):
    note_path = args.note
    text = open(note_path, encoding="utf-8").read()
    title, standfirst, provenance, body_md = parse_head(text)
    if not title:
        sys.exit("no level-one title on the first line of the note")

    repo_url, branch, note_rel = repo_context(note_path)
    if args.repo_url:
        repo_url = args.repo_url.rstrip("/")
    if args.branch:
        branch = args.branch
    if args.source_path:
        note_rel = args.source_path

    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
    body = md.convert(body_md)
    body = re.sub(r"^\s*<hr\s*/?>\s*", "", body, count=1)  # the head separator
    body = rewrite_md_links(body, repo_url, branch, note_rel)
    stand_html = rewrite_md_links(inline_html(standfirst), repo_url, branch, note_rel) if standfirst else ""
    prov_html = rewrite_md_links(inline_html(provenance), repo_url, branch, note_rel) if provenance else ""
    body = wrap_brief(body)
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
    body = mark_claims(body)

    figures_path = args.figures
    if not figures_path:
        cand = re.sub(r"\.md$", "", note_path) + ".figures.json"
        figures_path = cand if os.path.exists(cand) else None
    if figures_path:
        figures = json.load(open(figures_path, encoding="utf-8"))
        body = insert_figures(body, figures, os.path.dirname(os.path.abspath(figures_path)))

    toc = table_of_contents(body)

    date = ""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(note_path)) or re.search(r"(\d{4}-\d{2}-\d{2})", standfirst)
    if m:
        date = m.group(1)
    eyebrow = ["Reading note"]
    if date:
        eyebrow.append(date)
    if args.for_reader:
        eyebrow.append("Written for " + args.for_reader)
    project = args.project or (repo_url.rsplit("/", 1)[-1] if repo_url else "")
    if project:
        eyebrow.append(html.escape(project))

    if repo_url and note_rel:
        foot = (f'The same text is committed as <code>{html.escape(note_rel)}</code> in '
                f'<a href="{repo_url}">{html.escape(repo_url.split("github.com/")[-1])}</a>. '
                f'Where this page and that file differ, the file governs.')
    else:
        foot = (f'The same text is the file <code>{html.escape(os.path.basename(note_path))}</code>. '
                f'Where this page and that file differ, the file governs.')

    page = []
    page.append("<title>" + html.escape(args.title or title) + "</title>")
    page.append(FONTS)
    page.append("<style>" + CSS + "</style>")
    page.append('<div class="page">')
    page.append('<header class="mast">')
    page.append('<p class="eyebrow">' + "".join(f"<span>{e}</span>" for e in eyebrow) + "</p>")
    page.append("<h1>" + inline_html(title) + "</h1>")
    if stand_html:
        page.append('<p class="stand">' + stand_html + "</p>")
    if prov_html:
        page.append('<div class="prov"><span class="lbl">Provenance</span>' + prov_html +
                    '<div class="legend"><span>Claims are marked inline as</span>' +
                    mark("est", "established") + mark("inf", "inferred") + mark("spec", "speculation") + "</div></div>")
    page.append("</header>")
    if toc:
        page.append('<nav class="toc" aria-label="Sections"><p class="lbl">Sections</p><ol>' + toc + "</ol></nav>")
    page.append('<main class="prose">' + body + "</main>")
    page.append('<footer class="foot">' + foot + "</footer>")
    page.append("</div>")
    out_html = "\n".join(page)

    out = args.out or (re.sub(r"\.md$", "", os.path.basename(note_path)) + ".html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w", encoding="utf-8").write(out_html)
    written = [out]
    if args.preview:
        prev = re.sub(r"\.html$", "", out) + ".preview.html"
        open(prev, "w", encoding="utf-8").write(
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'
            + out_html + "</body></html>")
        written.append(prev)
    n_marks = out_html.count('class="mark ') - (3 if prov_html else 0)
    print(f"wrote {', '.join(written)}: {len(out_html)} bytes, {out_html.count('<h2 ')} sections, "
          f"{n_marks} marks, {out_html.count('<figure')} figures")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a designed HTML page from a reading note.")
    ap.add_argument("note", help="the markdown note")
    ap.add_argument("--out", help="output path (default: <note stem>.html in the current directory)")
    ap.add_argument("--title", help="short name for the browser tab (default: the note's title)")
    ap.add_argument("--repo-url", help="GitHub repository URL for rewriting relative .md links "
                    "(default: the note's origin remote)")
    ap.add_argument("--branch", help="branch for the rewritten links (default: origin's default branch)")
    ap.add_argument("--source-path", help="repository-relative path of the note for the footer "
                    "(default: from the git checkout)")
    ap.add_argument("--figures", help="sidecar JSON listing figures (default: <note>.figures.json if present)")
    ap.add_argument("--for", dest="for_reader", help='reader named in the eyebrow, e.g. "TC, the operator"')
    ap.add_argument("--project", help="project named in the eyebrow (default: the repository name)")
    ap.add_argument("--preview", action="store_true", help="also write a standalone .preview.html")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
