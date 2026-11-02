"""Basic checks of the supplied input files (not solution code).
Run from inside supplementary/:  python check_inputs.py   (needs numpy)
"""
import json
from pathlib import Path

import numpy as np

p = Path(__file__).resolve().parent
P = json.loads((p / "parameters.json").read_text())


def read(name):
    arr = np.genfromtxt(p / name, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.stack([arr["x"], arr["y"]], axis=1), np.atleast_1d(arr["group"]).astype(str)


T = P["q1_q2_q4"]
for tag, c in T["periodic_cells"].items():
    n = c["N_perfect"]
    for kind, count in (("perfect", n), ("vacancy", n - 1), ("interstitial", n + 1)):
        x, g = read(f"{kind}_{tag}.csv")
        assert len(x) == count and np.all(np.isfinite(x)) and np.all(g == "mobile")
    x, _ = read(f"perfect_{tag}.csv")
    L = np.array([c["Lx"], c["Ly"]])
    assert np.all(x >= 0) and np.all(x < L)
    d = x[:, None, :] - x[None, :, :]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=2)[np.triu_indices(len(x), 1)]
    assert abs(r.min() - T["lattice_spacing_a"]) < 1e-8
    print(f"{tag} (triangular): N = {n}, box = {L.round(6)}, nearest-neighbour distance = {r.min():.6f}")

Q = P["q3"]
x, g = read(Q["file"])
groups, counts = np.unique(g, return_counts=True)
assert len(x) == Q["N"]
print(f"{Q['file']} (square): {len(x)} atoms;", dict(zip(groups.tolist(), counts.tolist())))
print("Input checks passed.")
