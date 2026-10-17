"""Runs for half a minute: press Stop (Shift+F5) to end it, or let the time budget do so.

The traceback the stop leaves points at the line that was running.
"""

import time

for i in range(600):
    time.sleep(0.05)
    if i % 20 == 0:
        print("still running", i)
