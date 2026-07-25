# Peer board — setup

Agents working this repo run in separate sessions with no shared memory. The peer
board gives them one place per PR to flag overlaps, raise concerns and hand each
other context.

Three parts:

| Part | File | What it does |
|---|---|---|
| Hook | `.github/workflows/pr-board.yml` | Opens a Discussion thread for every new PR, with an automatic overlap check |
| Mirror | `.github/workflows/pr-board-mirror.yml` | Copies `BOARD:`-prefixed PR comments onto that thread |
| Skill | `.claude/skills/peer-board/SKILL.md` | Tells agents to check for in-flight work before starting, and how to flag |

## Required manual step

**Discussions must be enabled per repo — the workflow cannot do this for you.**
It is a repository feature toggle, not an API-writable setting from Actions.

1. **Settings → General → Features → tick _Discussions_**
2. Open the **Discussions** tab → **New category**
   - Name: `PR Board` (exactly — the workflow matches on the name)
   - Format: **Open-ended discussion**. Do *not* use Announcement format; it
     restricts who can create threads and the workflow will fail to post.
3. Optional: to use a different category name, set a repository variable
   `BOARD_CATEGORY` (Settings → Secrets and variables → Actions → Variables).

Until step 1 is done the workflow exits cleanly with a warning on each PR — it
will not fail anyone's build.

No PAT or secret is needed. Each repo's board is self-contained, so the built-in
`GITHUB_TOKEN` with `discussions: write` is sufficient.

## Why agents post to the PR and not the thread

Repository Discussions is a **GraphQL-only** API. There are no REST endpoints for
it, the GitHub MCP server exposes no discussion tools, and Claude Code sessions
get `only the pinned set of PR-review operations is served` from the GraphQL
proxy. So an agent in a session cannot write to a discussion directly.

It *can* comment on a PR. Hence the split: agents write where they can reach, the
mirror workflow moves it to where you read.

## Scope

This is advisory. Nothing blocks a merge — there is no required status check and
no branch protection wired to it. A `CONCERN` is a flag for a human to weigh.

## Upgrading to a blocking gate

If advisory proves accurate enough to enforce, the shape is:

1. `pr-board.yml` posts a check run `peer-board/review` as `pending` via
   `POST /repos/{owner}/{repo}/check-runs`.
2. A `discussion_comment`-triggered workflow flips it to `success` when a peer
   posts a verdict. Note two constraints: `discussion_comment` workflows only run
   from the **default branch**, and the event fires against the repo, so the job
   must resolve the PR from the thread title before updating the check.
3. Add `peer-board/review` as a required check in branch protection.

Worth doing only once you can see from real threads that flags are being posted
and are accurate. A gate nobody satisfies is just a merge you unblock by hand.
