"""Runs Q1-Q4. Usage: python run_all.py [q1 q2 q3 q4]   (default: all). Figures -> figs/, numbers -> results.json"""
import sys, time, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ms import *

OUT = Path(__file__).resolve().parent
FIG = OUT / "figs"; FIG.mkdir(exist_ok=True)
T = P["q1_q2_q4"]; A = T["lattice_spacing_a"]
RES = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
pot = Pot()


def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=150); plt.close()


def dump():
    (OUT / "results.json").write_text(json.dumps(RES, indent=1))


def cell(tag):
    c = T["periodic_cells"][tag]; return c, np.array([c["Lx"], c["Ly"]])


def window(X, centre, L, R):
    """atoms within R of centre (nearest image), coordinates shifted next to centre"""
    d = nearest(X - centre, L); m = np.linalg.norm(d, axis=1) < R
    return np.where(m)[0], centre + d[m]


def bonds(ax, pts, rc, **kw):
    r = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    for i, j in zip(*np.where(np.triu(r < rc, 1) & (r > 0))):
        ax.plot(*zip(pts[i], pts[j]), **kw)


def energies(tag):
    c, L = cell(tag); ev = Evaluator(pot, L)
    EN = ev.ef(read(f"perfect_{tag}.csv")[0])[0]
    return c, L, ev, EN


# ------------------------------------------------------------------ Q1
def q1():
    R = {}
    c, L, ev, EN = energies("20x20"); N = c["N_perfect"]
    Xp = read("perfect_20x20.csv")[0]; E, F = ev.ef(Xp)
    Xv = read("vacancy_20x20.csv")[0]; Ev0, Fv = ev.ef(Xv)
    R["a"] = dict(E_perfect=E, fmax_perfect=fmax_of(F), e0=E / N, E_vac=Ev0, fmax_vac=fmax_of(Fv))
    tol = 1e-5
    Xr, H = minimize(ev, Xv, tol=tol)
    Evr = H["E"][-1]
    Ef0 = Ev0 - (N - 1) / N * EN; Ef1 = Evr - (N - 1) / N * EN
    R["b"] = dict(tol=tol, iters=H["it"][-1], evals=H["nev"][-1], status=H["status"], E_final=Evr, fmax_final=H["fmax"][-1])
    R["c"] = dict(Ef_unrelaxed=Ef0, Ef_relaxed=Ef1, minus_e0=-E / N, relax_energy=Ef0 - Ef1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(H["it"], H["E"], "o-", ms=3); ax[0].set(xlabel="CG iteration", ylabel="total energy")
    ax[1].semilogy(H["it"], H["fmax"], "o-", ms=3); ax[1].axhline(tol, ls="--", c="k", lw=.7)
    ax[1].set(xlabel="CG iteration", ylabel="max atomic force")
    save("q1b_convergence.png")
    # displacements
    rd = np.array(c["removed_atom_position"])
    u = nearest(Xr - Xv, L)
    d = nearest(Xv - rd, L); r = np.linalg.norm(d, axis=1)
    ur = (u * d).sum(1) / r
    m = r < 4.2 * A
    s = round(0.35 * A / np.linalg.norm(u[m], axis=1).max(), -1)
    R["c"].update(max_disp=float(np.linalg.norm(u, axis=1).max()), arrow_factor=s,
                  ur_first_shell=float(ur[r < 1.1 * A].mean()), ur_second_shell=float(ur[(r > 1.5 * A) & (r < 1.9 * A)].mean()))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    P0 = rd + d[m]
    ax[0].quiver(P0[:, 0], P0[:, 1], s * u[m, 0], s * u[m, 1], angles="xy", scale_units="xy", scale=1, width=.005, color="C3")
    ax[0].scatter(*P0.T, s=14, c="k"); ax[0].plot(*rd, "x", c="b", ms=9)
    ax[0].set_aspect("equal"); ax[0].set(title=f"displacements x{s:g} (vacancy marked x)", xlabel="x", ylabel="y")
    ax[1].plot(r[m], ur[m], "o", ms=4); ax[1].axhline(0, c="k", lw=.5)
    ax[1].set(xlabel="distance from vacant site", ylabel="radial displacement $u_r$")
    save("q1c_displacement.png")
    # (d)
    R["d"] = {}
    for tag in ("10x10", "16x16", "20x20"):
        c, L, ev, EN = energies(tag); N = c["N_perfect"]
        Xv = read(f"vacancy_{tag}.csv")[0]; E0 = ev.energy(Xv)
        Xr, H = minimize(ev, Xv, tol=tol)
        R["d"][tag] = dict(N=N, Ef_unrelaxed=E0 - (N - 1) / N * EN, Ef_relaxed=H["E"][-1] - (N - 1) / N * EN,
                           iters=H["it"][-1], status=H["status"])
    RES["q1"] = R; dump(); print(json.dumps(R, indent=1))


# ------------------------------------------------------------------ Q2
def q2():
    R = {}
    c, L, ev, EN = energies("20x20"); N = c["N_perfect"]
    Xi = read("interstitial_20x20.csv")[0]
    E0, F0 = ev.ef(Xi); tol = 1e-5
    Xr, H = minimize(ev, Xi, tol=tol); E1 = H["E"][-1]
    R["a"] = dict(E_init=E0, fmax_init=fmax_of(F0), iters=H["it"][-1], evals=H["nev"][-1], E_final=E1, fmax_final=H["fmax"][-1], status=H["status"])
    R["b"] = dict(Ef_unrelaxed=E0 - (N + 1) / N * EN, Ef_relaxed=E1 - (N + 1) / N * EN)
    plt.figure(figsize=(4.5, 3.4)); plt.semilogy(H["it"], H["fmax"]); plt.xlabel("CG iteration"); plt.ylabel("max atomic force")
    plt.axhline(tol, ls="--", c="k", lw=.7); save("q2a_convergence.png")
    k = c["interstitial_atom_id"]; ri = Xi[k]; rf = Xr[k]
    u = nearest(Xr - Xi, L); un = np.linalg.norm(u, axis=1)
    # is the interstitial at the centre of its (original) triangle? distances to its 3 original nearest neighbours
    di = np.linalg.norm(nearest(Xi - Xi[k], L), axis=1); di[k] = 9
    nn = np.argsort(di)[:3]
    dist_f = np.linalg.norm(nearest(Xr[nn] - Xr[k], L), axis=1)
    R["c"] = dict(pos_init=ri.tolist(), pos_final=rf.tolist(), shift_of_interstitial=float(np.linalg.norm(nearest(rf - ri, L))),
                  dist_to_original_triangle_atoms_final=dist_f.tolist(),
                  max_disp_atom=float(un[np.arange(len(un)) != k].max()), max_disp_interstitial=float(un[k]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
    for a_, X_, ttl in ((ax[0], Xi, "before relaxation"), (ax[1], Xr, "after relaxation")):
        idx, pts = window(X_, X_[k] if a_ is ax[1] else ri, L, 3.2 * A)
        bonds(a_, pts, 1.37 * A, c="gray", lw=1)
        a_.scatter(*pts.T, s=30, c="C0"); j = list(idx).index(k); a_.scatter(*pts[j], s=90, c="r", zorder=3)
        a_.set_aspect("equal"); a_.set(title=ttl, xlabel="x", ylabel="y")
    save("q2c_structure.png")
    # row through the final position: the lattice row whose initial y is closest to the final y of the interstitial
    ys = Xi[:, 1]; rows = np.unique(np.round(ys, 5)); rowy = rows[np.argmin(np.abs(rows - rf[1]))]
    m = (np.abs(ys - rowy) < 1e-4); m[k] = False
    xr = nearest(Xr[m] - rf, L)[:, 0]  # position along row relative to interstitial
    o = np.argsort(xr)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(xr[o], u[m][o, 0], "o-", ms=3); ax.axhline(0, c="k", lw=.5)
    ax.set(xlabel="x - x of interstitial", ylabel="$u_x$ (atoms of the row)", title=f"row y = {rowy:.3f} (interstitial final y = {rf[1]:.3f})")
    save("q2c_row_ux.png")
    R["c"]["row_y"] = float(rowy); R["c"]["row_ux_min_max"] = [float(u[m, 0].min()), float(u[m, 0].max())]
    RES["q2"] = R; dump(); print(json.dumps(R, indent=1))


# ------------------------------------------------------------------ Q3
def q3():
    Q = P["q3"]; a = Q["lattice_spacing_a"]
    X0, g = read(Q["file"]); psq = Pot(sq=True); ev = Evaluator(psq, None, "verlet", 0.2)
    free = g == "mobile"; top = g == "top"; bot = g == "bottom"; tol = 1e-5
    ext = np.array(Q["extra_half_column_ids"])

    xmin, xmax = X0[:, 0].min(), X0[:, 0].max()
    ybt, ytb = X0[bot, 1].max(), X0[top, 1].min()

    def coord(X):
        r = np.linalg.norm(X[:, None] - X[None], axis=2)
        return ((r < 1.2 * a) & (r > 0)).sum(1)

    def core(X):
        c = coord(X)
        ok = free & (X[:, 0] > xmin + 2.5 * a) & (X[:, 0] < xmax - 2.5 * a) & (X[:, 1] > ybt + 1.5 * a) & (X[:, 1] < ytb - 1.5 * a)
        w = np.where(ok & (c != 4))[0]
        return w, c

    def structure(ax, X, title, w, c):
        pts = X; r = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        for i, j in zip(*np.where(np.triu(r < 1.2 * a, 1))):
            ax.plot(*zip(pts[i], pts[j]), c="lightgray", lw=.6, zorder=1)
        ax.scatter(*X[free | top | bot].T, s=5, c="k", zorder=2)
        ax.scatter(*X[ext].T, s=14, c="C2", zorder=3, label="extra half-column")
        ax.scatter(*X[w].T, s=40, facecolors="none", edgecolors="r", lw=1.3, zorder=4, label="coordination ≠ 4")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.legend(fontsize=6, loc="lower right")

    # (a)
    w0, c0 = core(X0)
    X1, H = minimize(ev, X0, free, tol=tol)
    w1, c1 = core(X1)
    R = dict(a=dict(E_init=H["E"][0], E_relaxed=H["E"][-1], iters=H["it"][-1], evals=H["nev"][-1], status=H["status"], fmax=H["fmax"][-1],
                    core_before_x=float(X0[w0, 0].mean()) if len(w0) else None, core_after_x=float(X1[w1, 0].mean()),
                    core_after_y=float(X1[w1, 1].mean()), n_core_atoms=len(w1), coord_values=sorted(set(c1[w1].tolist()))))
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    structure(ax[0], X0, "before relaxation", w0, c0); structure(ax[1], X1, "after relaxation", w1, c1)
    save("q3a_configuration.png")
    # (b)(c)(d)
    H_ = X1[top, 1].mean() - X1[bot, 1].mean(); W = np.ptp(X1[top, 0])
    R["geom"] = dict(H=H_, W=W)
    rows = []; X = X1.copy()
    def reaction(X):
        _, F = ev.ef(X); return -F[top, 0].sum()
    rows.append(dict(u=0.0, gamma=0.0, iters=H["it"][-1], fmax=H["fmax"][-1], E=H["E"][-1], Rx=reaction(X1),
                     core_x=float(X1[w1, 0].mean()), ncore=len(w1)))
    snaps = {0.0: (X1.copy(), w1, c1)}
    nsteps = int(round(Q["final_displacement"] / Q["displacement_increment"]))
    for n in range(1, nsteps + 1):
        u_ = n * Q["displacement_increment"]
        X = X.copy(); X[top, 0] = X1[top, 0] + u_
        X, Hn = minimize(ev, X, free, tol=tol)
        w, c = core(X)
        rows.append(dict(u=u_, gamma=u_ / H_, iters=Hn["it"][-1], fmax=Hn["fmax"][-1], E=Hn["E"][-1], Rx=reaction(X),
                         core_x=float(X[w, 0].mean()) if len(w) else float("nan"), ncore=len(w), status=Hn["status"]))
        snaps[round(u_, 4)] = (X.copy(), w, c)
        print(rows[-1])
    R["table"] = rows
    u = np.array([r["u"] for r in rows]); E = np.array([r["E"] for r in rows]); Rx = np.array([r["Rx"] for r in rows])
    cx = np.array([r["core_x"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(u, E - E[0], "o-", ms=3); ax[0].set(xlabel="u", ylabel="relaxed energy $-E(0)$")
    ax[1].plot(u, Rx, "o-", ms=3); ax[1].axhline(0, c="k", lw=.5); ax[1].set(xlabel="u", ylabel="$R_x$")
    for a_ in ax:
        a_.secondary_xaxis("top", functions=(lambda v: v / H_, lambda v: v * H_)).set_xlabel("$\\gamma$")
    save("q3c_energy_reaction.png")
    fig, ax = plt.subplots(figsize=(5, 3.6)); ax.plot(u, cx, "o-", ms=3); ax.set(xlabel="u", ylabel="core x"); save("q3d_core_x.png")
    # first move: first step where core x changes by more than a/2
    base = cx[0]; mv = np.where(np.abs(cx - base) > 0.5 * a)[0]
    i1 = int(mv[0]) if len(mv) else None
    d = dict(core_x0=base)
    if i1 is not None:
        d.update(u_first_move=float(u[i1]), u_before=float(u[i1 - 1]), Rx_before=float(Rx[i1 - 1]), tau_peierls=float(Rx[i1 - 1] / W),
                 Rx_max=float(Rx.max()), u_at_Rx_max=float(u[Rx.argmax()]))
    R["d"] = d
    # configuration at end + before the core leaves detection
    lost = np.where(np.isnan(cx))[0]
    keys = sorted(snaps)
    show = [keys[-1]] + ([round(u[lost[0]], 4)] if len(lost) else [])
    R["d"]["core_lost_at_u"] = float(u[lost[0]]) if len(lost) else None
    fig, ax = plt.subplots(1, len(show), figsize=(6 * len(show), 6), squeeze=False)
    for a_, kk in zip(ax[0], show):
        Xs, ws, cs = snaps[kk]; structure(a_, Xs, f"u = {kk}", ws, cs)
    save("q3d_final_configuration.png")
    # right surface step: x of rightmost atom in the top half vs bottom half of the mobile region, initial vs final
    Xf = snaps[keys[-1]][0]
    def edge(X, up):
        m = free & ((X[:, 1] > 0) if up else (X[:, 1] < 0)); return X[m, 0].max()
    R["d"]["right_edge_upper_minus_lower_initial"] = float(edge(X1, True) - edge(X1, False))
    R["d"]["right_edge_upper_minus_lower_final"] = float(edge(Xf, True) - edge(Xf, False))
    R["d"]["right_edge_upper_initial_minus_lower_final_shift"] = float(edge(Xf, True) - edge(X1, True))
    RES["q3"] = R; dump(); print(json.dumps({k: v for k, v in R.items() if k != "table"}, indent=1))


# ------------------------------------------------------------------ Q4
def q4():
    R = {}; tol = 1e-5
    def run(tag, mode, eps=0.2, kind="interstitial"):
        c, L = cell(tag); X = read(f"{kind}_{tag}.csv")[0]
        ev = Evaluator(pot, L, mode, eps)
        t = time.process_time(); Xr, H = minimize(ev, X, tol=tol); t = time.process_time() - t
        return ev, H, t
    R["b"] = {}
    for tag in ("10x10", "16x16", "20x20"):
        for mode in ("n2", "verlet"):
            ev, H, t = run(tag, mode)
            R["b"][f"{tag}_{mode}"] = dict(n_dist=ev.n_dist, cpu=t, builds=ev.n_build, E=H["E"][-1], iters=H["it"][-1], evals=ev.n_eval,
                                           list_len=len(ev.I) if mode == "verlet" else None, status=H["status"])
            print(tag, mode, R["b"][f"{tag}_{mode}"])
    R["c"] = {}
    for eps in (0.1, 0.2, 0.35):
        ev, H, t = run("20x20", "verlet", eps)
        R["c"][str(eps)] = dict(list_len=len(ev.I), builds=ev.n_build, n_dist=ev.n_dist, cpu=t, E=H["E"][-1], evals=ev.n_eval)
        print(eps, R["c"][str(eps)])
    R["d"] = {}; fig, ax = plt.subplots(figsize=(6, 3.8))
    for m_ in ("sd", "cg"):
        c, L = cell("20x20"); X = read("vacancy_20x20.csv")[0]; ev = Evaluator(pot, L, "verlet", 0.2)
        Xr, H = minimize(ev, X, method=m_, tol=tol, maxit=30000)
        R["d"][m_] = dict(iters=H["it"][-1], evals=H["nev"][-1], E=H["E"][-1], fmax=H["fmax"][-1], status=H["status"])
        ax.semilogy(H["nev"], H["fmax"], label=m_.upper()); print(m_, R["d"][m_])
    ax.axhline(tol, ls="--", c="k", lw=.6); ax.set(xlabel="energy/force evaluations", ylabel="max atomic force"); ax.legend()
    save("q4d_sd_vs_cg.png")
    RES["q4"] = R; dump(); print(json.dumps(R, indent=1))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["q1", "q2", "q3", "q4"]
    for q in todo:
        globals()[q]()
