"""2D molecular statics: LJ-type pair potentials, PBC, N2 / Verlet neighbour search, SD and CG.
Reduced LJ units. Set DATA to the folder with the input files."""
import json, math, os
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("MS_DATA", Path(__file__).resolve().parent / "data"))
P = json.loads((DATA / "parameters.json").read_text())
RC = P["r_cut"]


def read(name):
    a = np.genfromtxt(DATA / name, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.stack([a["x"], a["y"]], 1), np.atleast_1d(a["group"]).astype(str)


class Pot:
    """phi_LJ (optionally minus Gaussian well), truncated with cutoff correction (Eq. 2)."""
    def __init__(s, sq=False):
        s.sq = sq
        q = P["q3"]
        s.e2, s.r2, s.s = q["eps2"], q["r2"], q["s"]
        s.pc, s.dpc = s.raw(RC)

    def raw(s, r):
        ir6 = r ** -6
        phi = 4 * (ir6 * ir6 - ir6)
        dphi = 24 * (r ** -7 - 2 * r ** -13)
        if s.sq:
            g = np.exp(-(r - s.r2) ** 2 / (2 * s.s ** 2))
            phi = phi - s.e2 * g
            dphi = dphi + s.e2 * (r - s.r2) / s.s ** 2 * g
        return phi, dphi

    def __call__(s, r):
        phi, dphi = s.raw(r)
        return phi - s.pc - (r - RC) * s.dpc, dphi - s.dpc


def nearest(d, L):
    return d if L is None else d - L * np.round(d / L)


_tri = {}
def triu(N):
    if N not in _tri:
        _tri[N] = np.triu_indices(N, 1)
    return _tri[N]


class Evaluator:
    """Energy/force with mode 'n2' or 'verlet'. Counts pair distances, evaluations, list builds."""
    def __init__(s, pot, L=None, mode="n2", eps=0.2):
        s.pot, s.L, s.mode, s.eps = pot, L, mode, eps
        s.rneigh = (1 + eps) * RC
        if L is not None:
            assert s.rneigh <= L.min() / 2 or mode == "n2" and RC <= L.min() / 2
        s.n_dist = s.n_eval = s.n_build = 0
        s.X0 = None

    def reset_counts(s):
        s.n_dist = s.n_eval = s.n_build = 0; s.X0 = None

    def _build(s, X):
        N = len(X); I, J = triu(N)
        d = nearest(X[I] - X[J], s.L)
        m = (d * d).sum(1) < s.rneigh ** 2
        s.I, s.J, s.X0 = I[m], J[m], X.copy()
        s.n_dist += N * (N - 1) // 2; s.n_build += 1

    def _pairs(s, X):
        N = len(X)
        if s.mode == "n2":
            I, J = triu(N); s.n_dist += N * (N - 1) // 2
            return I, J
        if s.X0 is None:
            s._build(X)
        else:
            dn = np.linalg.norm(nearest(X - s.X0, s.L), axis=1)
            d1, d2 = np.partition(dn, -2)[-2:]
            if d1 + d2 > s.eps * RC:   # two largest displacements: no pair can close from r_neigh to r_cut
                s._build(X)
        s.n_dist += len(s.I)
        return s.I, s.J

    def energy(s, X):
        s.n_eval += 1
        I, J = s._pairs(X)
        r = np.linalg.norm(nearest(X[I] - X[J], s.L), axis=1)
        return math.fsum(s.pot(r[r < RC])[0])

    def ef(s, X):
        s.n_eval += 1
        I, J = s._pairs(X)
        d = nearest(X[I] - X[J], s.L)
        r = np.linalg.norm(d, axis=1)
        m = r < RC
        I, J, d, r = I[m], J[m], d[m], r[m]
        phi, dphi = s.pot(r)
        f = -(dphi / r)[:, None] * d          # force on I
        N = len(X)
        F = np.stack([np.bincount(I, f[:, k], N) - np.bincount(J, f[:, k], N) for k in (0, 1)], 1)
        return math.fsum(phi), F


def fmax_of(F, mask=None):
    n = np.linalg.norm(F, axis=1)
    return n.max() if mask is None else n[mask].max()


C1 = 1e-4   # sufficient-decrease (Armijo) constant


def linemin(ev, X, E0, d, s, slope):
    """Bracket (first step set through a length s) + parabolic interpolation; the accepted step must satisfy
    the sufficient-decrease test E(a) <= E0 - C1*a*slope, slope = f.d > 0."""
    ok = lambda a, E: E <= E0 - C1 * a * slope
    dm = np.linalg.norm(d, axis=1).max()
    a1 = s / dm
    E1 = ev.energy(X + a1 * d)
    while not ok(a1, E1):
        a1 /= 2
        if a1 * dm < 1e-13:
            return None
        E1 = ev.energy(X + a1 * d)
    a2 = 2 * a1
    E2 = ev.energy(X + a2 * d)
    while E2 <= E1:
        a1, E1 = a2, E2
        a2 *= 2; E2 = ev.energy(X + a2 * d)
    c = ((E2 - E0) / a2 - (E1 - E0) / a1) / (a2 - a1)
    b = (E1 - E0) / a1 - c * a1
    am = -b / (2 * c) if c > 0 else a1
    am = min(max(am, 1e-3 * a1), a2)
    Em = ev.energy(X + am * d)
    a, E = (am, Em) if Em < E1 else (a1, E1)
    while not ok(a, E):                       # back off until sufficient decrease holds
        a /= 2
        if a * dm < 1e-13:
            return None
        E = ev.energy(X + a * d)
    return a


def minimize(ev, X0, free=None, method="cg", tol=1e-5, maxit=50000):
    """CG (Polak-Ribiere+, reset if f.d<=0) or steepest descent. Forces of fixed atoms set to zero.
    Returns X, history dict (iteration, energy, fmax, nevals)."""
    X = X0.copy(); N = len(X)
    mask = np.ones(N, bool) if free is None else free
    ev.n_eval0 = ev.n_eval
    E, F = ev.ef(X); F[~mask] = 0
    d = F.copy(); s = 0.01
    H = dict(it=[0], E=[E], fmax=[fmax_of(F)], nev=[ev.n_eval - ev.n_eval0])
    it = 0; status = "converged"
    while H["fmax"][-1] >= tol:
        if it >= maxit:
            status = "maxit"; break
        a = None
        for attempt in range(2):
            a = linemin(ev, X, E, d, s, np.sum(F * d))
            if a is not None:
                break
            d = F.copy()                        # retry once along steepest descent
        if a is None:
            status = "linemin failed"; break
        Xn = X + a * d
        En, Fn = ev.ef(Xn); Fn[~mask] = 0
        if En >= E:
            status = "no decrease"; break
        s = 2 * a * np.linalg.norm(d, axis=1).max()
        if method == "cg":
            beta = max(0.0, np.sum(Fn * (Fn - F)) / np.sum(F * F))
            d = Fn + beta * d
            if np.sum(Fn * d) <= 0:
                d = Fn.copy()
        else:
            d = Fn.copy()
        X, E, F = Xn, En, Fn
        it += 1
        H["it"].append(it); H["E"].append(E); H["fmax"].append(fmax_of(F)); H["nev"].append(ev.n_eval - ev.n_eval0)
    H["status"] = status
    return X, H
