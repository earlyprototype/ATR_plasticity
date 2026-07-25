---
name: peer-board
description: Check what other agents are already working on before starting, and flag overlaps or concerns on their PRs. Use at the START of any task that will produce a commit, spec, experiment or PR in this repo — and whenever asked to review, comment on, or coordinate around someone else's PR. Triggers on "start work on issue N", "run experiment", "register a spec", "open a PR", "review PR N", "is anyone else working on this", "check for duplicate work", "what's in flight".
---

# Peer board

Several agents work these repos in parallel, in separate sessions, with no shared
memory. Without a deliberate check, they duplicate each other's work and collide
on identifiers. This has already happened here: issue #7 in `ATR_research` drew
three independent PRs (#8, #9, #19) implementing the same permutation test, two of
which disagreed on whether `W_U` is a distinct space from `W_E`; issues #5, #10 and
#20 all claimed hypothesis ID **H11**.

This skill is the check that prevents that. Two duties: **look before you start**,
and **flag what you can see that others cannot**.

## Before you start work

Do this before writing a spec, running an experiment, or touching a file. It costs
one or two tool calls and is the whole point of the skill.

1. **Is this issue already claimed?** List open PRs and look for any that reference
   your issue number in the body:

   - `mcp__github__list_pull_requests` (state `open`), then scan bodies for `#N`
   - or `mcp__github__search_issues` with `repo:OWNER/REPO is:pr is:open N`

   If another open PR claims your issue, **stop and read it**. Then either pick up
   where it left off, or post a `BOARD: DUPLICATE` on it saying you are taking a
   different angle and what that angle is. Do not silently start a second one.

2. **Is your identifier free?** Hypothesis and experiment IDs collide across
   branches because each agent only sees its own. Before assigning `H<n>` or
   `EXP_<id>`, check every branch, not just yours:

   ```bash
   git fetch origin '+refs/heads/*:refs/remotes/origin/*'
   git grep -hoE '\bH[0-9]+[a-z]?\b' $(git for-each-ref --format='%(refname)' refs/remotes/origin) -- '*SPEC*.md' '*RESULTS*.md' | sort -u
   ```

   Open PRs can hold IDs that are not yet on any branch you have — check their
   bodies too. If the next free number is ambiguous, say so explicitly in your
   spec rather than picking one and hoping.

3. **Has this already been answered?** Search merged PRs and results records
   before running anything expensive. A control that was run and recorded three
   weeks ago on another branch is still a run.

## When you open a PR

A board thread opens automatically for every new PR (`.github/workflows/pr-board.yml`)
and posts back a link. The overlap detection in that thread reads your PR body, so
**declare claims in a form it can parse**:

- `Closes #7` / `Fixes #7` / `Resolves #7` — for work that completes the issue
- `issue #7` — for work that advances it without closing it

A PR that describes its issue only in prose gets no overlap check.

## Flagging someone else's work

You cannot post to Discussions directly — the Discussions API is GraphQL-only and
sessions have no discussion tools. Instead comment **on the PR** with
`mcp__github__add_issue_comment`, starting the body with `BOARD:`. A mirror
workflow copies it onto the board thread.

| Prefix | Use it for |
|---|---|
| `BOARD: CONCUR` | Read it, no objection. Worth posting — it makes silence unambiguous |
| `BOARD: CONTEXT` | Something the author could not have known: a prior run, a related result, a spec that already covers this |
| `BOARD: CONCERN` | A methodological worry that should be answered before merge |
| `BOARD: DUPLICATE` | Repeats work done or in flight — always name the PR, commit or results section |
| `BOARD: COLLISION` | Identifier clash: hypothesis number, experiment ID, spec filename |

Write flags so they are actionable without the reader scrolling back:

> `BOARD: CONCERN` — This PR treats `W_U` as distinct from `W_E` and runs two tests
> per set. PR #19 states the gpt2-medium checkpoint carries `wte.weight` only, so
> the weights are tied and the two spaces are identical by identity. If that is
> right, the Bonferroni correction here is over 20 tests when it should be 10, and
> the headline off-band result is a duplicate of a test already counted. Worth
> settling before either lands.

Not:

> `BOARD: CONCERN` — possible issue with the embedding space, please check.

## What this does and does not do

**Advisory.** No flag blocks a merge. There is no required status check and no
branch protection wired to this. A `CONCERN` is a record for TC to weigh.

Because it does not enforce, the value is entirely in whether flags get posted at
all. If you read another agent's PR and have nothing to add, post `CONCUR` — an
empty board is indistinguishable from an unread one.

**Do not** use the board to relitigate a decision TC has already made, or to
re-run a peer's analysis just to check it. Flag what you noticed in passing; the
point is cheap early warning, not a second review layer.
