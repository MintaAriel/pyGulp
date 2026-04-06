#!/usr/bin/env python3
import argparse
from pygulp.forcefields.gnff_fine_tun import tune_gfnff


def main():
    ap = argparse.ArgumentParser(description="Compare ase operators")
    ap.add_argument("--delta", type=float, default=0.1, help="delta for parameters from the og gfnff")
    ap.add_argument("--name", type=str, default="tunning", help="db_name")
    ap.add_argument("--trials", type=int, default=100, help="number of trials for Bayesian opt")
    ap.add_argument("--descriptor", type=str, default="soap", help="ddescriptor for molecular crystal")

    args = ap.parse_args()

    tune_gfnff(delta_par=args.delta,
               db_name=args.name,
               n_trials=args.trials,
               fingerprint=args.descriptor)

if __name__ == "__main__":
    main()
