# pyGULP

`pyGULP` is a Python module for working with GULP through the Atomic Simulation Environment (ASE) calculator interface.

The package extends the standard GULP calculator with additional tools for molecular crystal simulations, structure optimisation, and analysis.

## Features

The module provides several utilities that simplify workflows with GULP:

- Analyse GULP optimisation runs and visualise them using ASE trajectory files, including bond length changes at each optimisation step.
- Optimise GFN-FF parameters for molecular crystals using experimental structures as reference.
- Parse GULP input files, including internal derivatives and cell parameters at pressures \( P > 0 \).
- Perform relaxation of molecular crystal structures.
- Add custom optimisation tags and constraints.

## Supported Force Fields

The current implementation supports several force-field models:

- ReaxFF  
- Dreiding  
- GFN-FF  

## Optimisation Utilities

Additional routines are included for rigid or symmetry-constrained optimisation of molecular crystals.

These routines can work both with standard GULP force fields and with external models such as UMA (Universal Model of Atoms).

## Project Structure

```
pyGULP
├── src/        main Python package
├── scripts/    helper scripts and small workflows
├── tests/      benchmarks and example calculations
├── data/       input structures (e.g. CIF files)
└── results/    generated output databases (ignored by git)
```

The main package is located in:

```
src/pygulp
```

and contains modules for:

- GULP input/output handling
- structure manipulation
- force-field optimisation
- relaxation routines
- analysis of optimisation trajectories

## Notes

This project is mainly intended for molecular crystal optimisation workflows where GULP is used together with ASE-based tools.

