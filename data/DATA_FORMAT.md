# Assignment 2: input files

All quantities are in reduced Lennard-Jones units (epsilon = sigma = 1).

## Files

| File | Question | Content |
|---|---|---|
| `perfect_10x10.csv`, `perfect_16x16.csv`, `perfect_20x20.csv` | Q1, Q4 | perfect triangular crystal |
| `vacancy_10x10.csv`, `vacancy_16x16.csv`, `vacancy_20x20.csv` | Q1, Q4 | perfect crystal with one atom removed |
| `interstitial_10x10.csv`, `interstitial_16x16.csv`, `interstitial_20x20.csv` | Q2, Q4 | perfect crystal with one extra atom |
| `edge_dislocation_initial.csv` | Q3 | square-lattice block containing an edge dislocation |
| `parameters.json` | all | lattice spacings, box sizes, defect positions, Q3 data |
| `check_inputs.py` | – | checks that the files are complete |
| `supplementary_note.pdf` | – | background material |

## CSV format

Each line is one atom: `id, x, y, group`.

- `id` starts at 0.
- `group` is `mobile` for every atom, except in `edge_dislocation_initial.csv`.

## Triangular cells (Q1, Q2, Q4)

- Periodic in x and y. The box runs from (0, 0) to (Lx, Ly).
- Take `Lx` and `Ly` from `parameters.json`. Do not compute them from a rounded value of a: a box that is wrong by 10⁻⁶ already puts spurious forces of about 10⁻⁴ on the perfect crystal.
- In the interstitial files the extra atom is the last line.
- The vacancy site and the starting position of the extra atom are also given in `parameters.json`.

## Dislocation block (Q3)

- 1022 atoms, square lattice, no periodic boundaries.
- `group` column:
  - `bottom`: two lowest rows, held fixed;
  - `top`: two highest rows, held fixed and displaced during loading;
  - `mobile`: all other atoms.
- The left and right surfaces are free.
- The last 14 lines (ids 1008 to 1021) are the extra half-column.
- The coordinates are not relaxed.

## Checking the files

```
python check_inputs.py
```

This needs NumPy. It checks the number of atoms, the box and the nearest-neighbour distance.
