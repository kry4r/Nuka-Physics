import argparse

from .runner import run

parser = argparse.ArgumentParser(prog="python -m nuka.train", description="Runs the training job of a run folder.")
parser.add_argument("job", help="the folder's job.json")
raise SystemExit(run(parser.parse_args().job))
