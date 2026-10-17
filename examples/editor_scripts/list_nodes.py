"""Counts the open scene's nodes by kind, then selects and frames the first body.

Open it in the Script panel and press F5; ``nodes``, ``select`` and ``action`` are bound in
every editor script.
"""

from collections import Counter

found = nodes()
print(f"{len(found)} nodes")
for kind, count in Counter(n["kind"] for n in found).most_common():
    print(f"  {kind:<8} {count}")

bodies = nodes(kind="body")
if bodies:
    select(bodies[0]["path"])
    action("view.frame_selected")
    print("selected", bodies[0]["path"])
