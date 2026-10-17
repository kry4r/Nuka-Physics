"""Drops a ring of boxes onto the scene, one undoable spawn each; Ctrl+Z takes them back.

Each spawn is the same scene.spawn action the Add menu runs.
"""

import math

COUNT = 8
RADIUS = 1.5
HEIGHT = 1.0

for i in range(COUNT):
    a = 2.0 * math.pi * i / COUNT
    pose = {"pos": [RADIUS * math.cos(a), RADIUS * math.sin(a), HEIGHT], "quat": [1.0, 0.0, 0.0, 0.0]}
    action("scene.spawn", kind="box", pose=pose)
print(f"dropped {COUNT} boxes; press Space to play")
