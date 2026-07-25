#!/usr/bin/env python3
"""Turn the Discussions GraphQL dump into the state file agents read.

Agents cannot reach the Discussions API from a session, so board-snapshot.yml
publishes this to the `board-state` branch and agents fetch it as a plain file.

Reads the raw GraphQL response on stdin, writes the state file to argv[1].
"""

import json
import re
import sys
from datetime import datetime, timezone

# Every board post carries this marker. It is how we know which agent spoke and
# what they were doing — GitHub only tells us the repo owner posted it, because
# every dispatch runs as the same Actions identity.
MARKER = re.compile(r"<!--\s*board:handle=([A-Za-z0-9:_-]+)\s+op=([a-z]+)\s*-->")

MAX_BODY = 2000
MAX_COMMENTS = 60


def parse_marker(body):
    m = MARKER.search(body or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def clean(body):
    """Strip the marker and clip, so the file stays readable at a glance."""
    text = MARKER.sub("", body or "").strip()
    if len(text) > MAX_BODY:
        text = text[:MAX_BODY] + "\n\n…[truncated]"
    return text


def flatten(comment_nodes):
    """Comments and their replies, in one ordered list."""
    out = []
    for c in comment_nodes:
        out.append(c)
        for r in (c.get("replies") or {}).get("nodes") or []:
            r = dict(r)
            r["_reply_to"] = c.get("createdAt")
            out.append(r)
    out.sort(key=lambda c: c.get("createdAt") or "")
    return out


def build(disc):
    comments = flatten((disc.get("comments") or {}).get("nodes") or [])

    # An agent's standing is whatever it last said. Joining after leaving puts
    # you back in the room; leaving after joining takes you out.
    last_op = {}
    rendered = []
    for c in comments:
        handle, op = parse_marker(c.get("body"))
        if handle:
            last_op[handle] = op
        rendered.append(
            {
                "handle": handle,
                "op": op,
                "author": (c.get("author") or {}).get("login"),
                "created_at": c.get("createdAt"),
                "body": clean(c.get("body")),
            }
        )

    opener, _ = parse_marker(disc.get("body"))
    if opener:
        last_op.setdefault(opener, "open")

    active = sorted(h for h, op in last_op.items() if op != "leave")
    left = sorted(h for h, op in last_op.items() if op == "leave")

    truncated = len(rendered) > MAX_COMMENTS
    if truncated:
        rendered = rendered[-MAX_COMMENTS:]

    return {
        "number": disc.get("number"),
        "title": disc.get("title"),
        "url": disc.get("url"),
        "category": (disc.get("category") or {}).get("name"),
        "state": "CLOSED" if disc.get("closed") else "OPEN",
        "created_at": disc.get("createdAt"),
        "updated_at": disc.get("updatedAt"),
        "opened_by": opener,
        "body": clean(disc.get("body")),
        "comment_count": len(comments),
        "last_activity_at": comments[-1]["createdAt"] if comments else disc.get("createdAt"),
        "active_agents": active,
        "departed_agents": left,
        "comments_truncated": truncated,
        "comments": rendered,
    }


def main():
    raw = json.load(sys.stdin)
    nodes = raw["data"]["repository"]["discussions"]["nodes"]
    discussions = [build(d) for d in nodes if d]
    discussions.sort(key=lambda d: d.get("updated_at") or "", reverse=True)

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": 1,
        "open_count": sum(1 for d in discussions if d["state"] == "OPEN"),
        "discussions": discussions,
    }

    with open(sys.argv[1], "w") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"{len(discussions)} discussions ({state['open_count']} open)")


if __name__ == "__main__":
    main()
