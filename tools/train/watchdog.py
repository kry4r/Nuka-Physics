"""Training watchdog: poll logs, exit on completion/crash/stagnation.

Exit codes / last line: WATCH[<TASK> COMPLETED|CRASHED|STAGNANT] ...
STAGNANT = reward flat (no +0.5 improvement) for STAG_MINUTES while frames
advance -- the signal to stop training early and validate.
"""
import re
import sys
import time

POLLS = 60           # seconds between samples
STAG_SAMPLES = 12    # consecutive non-improving samples -> stagnation

TASKS = {
    "HSWALK": "logs/train_hs_walk.log",
    "BFPD3": "logs/train_backflip_pd.log",
}


def read_task(path):
    txt_tail = ""
    try:
        with open(path, "r", errors="ignore") as f:
            txt_tail = f.read()[-200_000:]
    except OSError:
        return None
    m_ep = re.findall(r"epoch: (\d+) frames: (\d+)/40000000", txt_tail)
    m_rw = re.findall(r"rewards:\s+\[([-0-9.e]+)", txt_tail)
    done = "MAX FRAMES NUM!" in txt_tail[-20_000:]
    crash = "Traceback" in txt_tail[-10_000:]
    if not m_ep:
        return None
    ep, frames = int(m_ep[-1][0]), int(m_ep[-1][1])
    rew = float(m_rw[-1]) if m_rw else 0.0
    return dict(ep=ep, frames=frames, rew=rew, done=done, crash=crash)


def main():
    best = {k: 0.0 for k in TASKS}
    stag = {k: 0 for k in TASKS}
    seen = {k: None for k in TASKS}   # last DISTINCT rewards value observed
    while True:
        for name, path in TASKS.items():
            st = read_task(path)
            if st is None:
                continue
            ts = time.strftime("%H:%M:%S")
            if st["done"]:
                print(f"WATCH[{name} COMPLETED] ep={st['ep']} frames={st['frames']} rew={st['rew']}")
                return
            if st["crash"]:
                print(f"WATCH[{name} CRASHED] ep={st['ep']} frames={st['frames']}")
                return
            # Only judge trend on NEW distinct reward prints -- the same log line
            # re-read between print_stats windows says nothing about progress.
            if st["rew"] != seen[name]:
                seen[name] = st["rew"]
                if st["rew"] > best[name] + 0.5:
                    best[name] = st["rew"]
                    stag[name] = 0
                    line = "new-best"
                else:
                    stag[name] += 1
                    line = f"stag={stag[name]}/{STAG_SAMPLES}"
                with open("logs/watchdog.log", "a") as f:
                    f.write(f"{ts} WATCH[{name} OK] ep={st['ep']} frames={st['frames']} "
                            f"rew={st['rew']} {line}\n")
                if stag[name] >= STAG_SAMPLES:
                    print(f"WATCH[{name} STAGNANT] ep={st['ep']} frames={st['frames']} "
                          f"rew={st['rew']} ({STAG_SAMPLES} fresh prints, none beat best)")
                    return
        time.sleep(POLLS)


if __name__ == "__main__":
    sys.exit(main())
