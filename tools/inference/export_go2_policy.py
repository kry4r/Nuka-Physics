#!/usr/bin/env python3
"""Export an rl_games Go2 actor (+ input normalizer) to a flat binary the C++
MlpPolicy runner loads, plus a binary golden of (obs -> mu) pairs computed by
torch for the C++-vs-Python numeric validation (max|delta| < 1e-4).

The format and the reference forward live in nuka.train.export.

Usage:
  export_go2_policy.py <checkpoint.pth> <out_weights.bin> [--golden out_golden.bin]
                       [--samples N]
"""
import argparse
import sys

from nuka.train.export import load_state_dict, write_golden, write_nkml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("weights_out")
    ap.add_argument("--golden", default="")
    ap.add_argument("--samples", type=int, default=256)
    args = ap.parse_args()

    sd = load_state_dict(args.checkpoint)
    obs_dim, act_dim = write_nkml(args.weights_out, sd)
    print(f"wrote {args.weights_out}  obs_dim={obs_dim} act_dim={act_dim}")
    if args.golden:
        write_golden(args.golden, sd, obs_dim, act_dim, args.samples)
        print(f"wrote {args.golden}  samples={args.samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
