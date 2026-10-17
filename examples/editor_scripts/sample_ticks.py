"""Samples the world at the tick boundary: every 60 ticks it prints the count and the state
hash, and after five samples it takes its callback off. Run it, then press Space to play.

The callback runs on the physics thread, where ``app.world`` may be read directly.
"""

EVERY = 60
SAMPLES = 5
seen = {"ticks": 0, "samples": 0}


def sample(app):
    seen["ticks"] += 1
    if seen["ticks"] % EVERY:
        return
    seen["samples"] += 1
    print(f"tick {seen['ticks']:>5}  state {app.world.state_hash():016x}")
    if seen["samples"] == SAMPLES:
        app.off_tick(sample)
        print("sampling done")


app.on_tick(sample)
print(f"sampling every {EVERY} ticks; press Space to play")
