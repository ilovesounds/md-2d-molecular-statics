# 2D Molecular Statics: vacancy, interstitial, edge dislocation, neighbour lists
Assignment 2, Multiscale Modelling and Computation (APL410/AML7200). All quantities are in reduced Lennard-Jones units (epsilon = sigma = 1). Everything is written from scratch with NumPy (no LAMMPS, no library minimizers).

```
pip install -r requirements.txt
python run_all.py            # all questions  (about 2-3 min);  or: python run_all.py q1 q3
```
Outputs: `figs/*.png` (all plots) and `results.json` (all numbers quoted below). Input files live in `data/` (`MS_DATA=/other/path` overrides).

| File | Role |
|---|---|
| `ms.py` | the engine: potentials, periodic boundaries, N2 and Verlet neighbour search, energy/forces, line minimization, CG and steepest descent |
| `run_all.py` | one function per question (`q1`..`q4`): sets up the configuration, calls the engine, makes the plots, stores the numbers |
| `data/` | supplied inputs, assignment, supplementary note |

---
## 0. What the whole assignment is about
**Molecular statics** = find the zero-temperature atomic arrangement by minimizing the potential energy V(r_1..r_N) over all atom positions. No time, no velocities: you start from a guess and slide downhill until all forces vanish. Crystal defects (a missing atom, an extra atom, a dislocation) are exactly the situations where a guess is not the answer, so relaxation shows how the lattice distorts around them and what they cost in energy. This is the atomistic side of multiscale modelling: defect energies and the stress to move a dislocation are the inputs that continuum/crystal-plasticity models need.

### Physical model (shared by all questions) - `Pot` in `ms.py`
* Atoms interact in pairs through Lennard-Jones, phi_LJ(r) = 4(r^-12 - r^-6): strong repulsion at short range, weak attraction at long range.
* A plain cutoff leaves a jump in energy and force at r_cut = 2.5, which ruins minimization. The **cutoff correction** (Eq. 2) subtracts phi_LJ(r_cut) and the linear term (r - r_cut)phi_LJ'(r_cut), so both phi and phi' are zero at r_cut. Forces use phi'(r) - phi_LJ'(r_cut): the derivative of the correction term must be included, otherwise force and energy disagree.
* Total energy V = sum over pairs phi(r); force on atom a is -dV/dr_a, accumulated over all pairs (`Evaluator.ef`, using `bincount`).
* **Periodic boundary conditions (PBC)**: the cell is repeated infinitely, so a bulk crystal is mimicked by a small box. The distance between two atoms uses the nearest image, d -= L*round(d/L) (`nearest`). This is valid only if the interaction range <= L/2 (asserted in code).
* The lattice spacing a = 1.1161533 is the one that minimizes the energy of the *truncated* crystal, so the perfect crystal has (almost) zero force. The box lengths come from `parameters.json` exactly; a box wrong by 1e-6 creates spurious forces ~1e-4.

### The minimizer - `minimize`, `linemin` in `ms.py`
1. **Direction.** Steepest descent (SD): d = f. Conjugate gradient (CG): d = f + beta d_old with Polak-Ribiere beta = f_new.(f_new - f_old)/(f_old.f_old), clipped at 0; if the new direction is not downhill (f.d <= 0) reset d = f. CG remembers previous directions so it does not zig-zag in long narrow energy valleys, which is why it needs far fewer iterations than SD.
2. **Line minimization** along d: choose a first trial step so the fastest atom moves a length s (0.01 first, then twice the previous move), halve until energy decreases, double while it keeps decreasing (bracketing), then fit a parabola through three points to get the step. The step must satisfy the **sufficient-decrease (Armijo) test** E(a) <= E(0) - 1e-4 a (f.d), otherwise it is halved.
3. **Stop** when the largest force magnitude over mobile atoms is below tol = 1e-5. Energies are summed with `math.fsum` because near convergence energy changes (~1e-12) are at the round-off level of a total energy of order 1e3.
4. **Constraints.** Fixed atoms get zero force inside the minimizer (`free` mask), so they never move.

### Neighbour search - `Evaluator`
* **N2 search**: check all N(N-1)/2 pairs every evaluation; simple, cost grows as N^2.
* **Verlet list**: once in a while compute all pairs within r_neigh = (1+eps) r_cut, then evaluate only those. Rebuild when delta_max1 + delta_max2 > eps r_cut (the two largest displacements since the last build). Safe because two atoms can approach each other by at most the sum of their displacements, which is at most the sum of the two largest ones; so a pair that was beyond r_neigh at build time cannot be within r_cut before the criterion triggers.
* The evaluator counts every pair distance (also during builds), evaluations and builds, for Q4.

---
## Q1. Vacancy formation and relaxation  (`q1`)
**Context.** Vacancies (missing atoms) are the most common point defect; they control diffusion and creep. Key quantity: the formation energy E_f = E_{N-1}(vacancy cell) - (N-1)/N E_N(perfect), i.e. the energy cost compared with taking that atom out and putting it back in the bulk (energy e0 = E_N/N per atom).
**Code.** Read `perfect_/vacancy_20x20.csv`; (a) evaluate energy and max force of both; (b) `minimize` the vacancy with CG, log energy and max force per iteration; (c) compute E_f before/after relaxation, displacement u = r_relaxed - r_initial (with nearest image), arrows magnified by a stated factor, radial displacement u_r = u.(r - r_vac)/|r - r_vac| against distance; (d) repeat for 10x10 and 16x16.
**Results / explanation.**
* Perfect crystal: E = -1164.6856 (e0 = -2.9117), max force 1.4e-8, i.e. zero as it should.
* Unrelaxed E_f = 2.91171 = -e0 exactly. With a pair potential each bond is shared by two atoms, so removing an atom deletes bonds worth 2 e0 in total, giving E_f = -e0 = the cohesive energy. Real metals have E_f about a third of that, because in many-body (EAM) models the other atoms' bonds strengthen when one neighbour is missing; a pair potential cannot do that.
* Relaxed E_f = 2.9079; relaxation is tiny (0.0038, max displacement 0.0032, 30 CG iterations) because the missing atom's bonds were only weakly loaded. Neighbours move *toward* the vacancy (u_r < 0).
* Cell size: 2.90792 (10x10), 2.90790 (16x16), 2.90790 (20x20); the defect's displacement field decays quickly so the image interaction is negligible.

## Q2. Interstitial  (`q2`)
**Context.** Opposite point defect: an extra atom squeezed into the lattice, placed at the centre of a triangle of three neighbours (only about 0.64 away from each, on the r^-12 wall).
**Code.** Evaluate, relax with CG, E_f = E_{N+1} - (N+1)/N E_N before/after, plot atoms with first-neighbour bonds (distance < 1.37a) before and after, and u_x along the close-packed row through the final position.
**Results / explanation.**
* Start: E = +1005.5, max force 1.4e4 (huge repulsion). After 413 CG iterations E = -1153.4038.
* E_f = 2173 unrelaxed, **14.19 relaxed**, about 5 times the vacancy. The inserted atom does not stay at the triangle centre: it moves 0.32 into the close-packed row and the row accommodates one extra atom, u_x antisymmetric about it with peaks of about +/-0.45. The compression is spread along the row over many bonds, which is cheapest for a very steep repulsive wall. Many bonds stay compressed, hence E_f(interstitial) >> E_f(vacancy), the same order as in real metals.

## Q3. Edge dislocation under quasi-static shear  (`q3`)
**Context.** Dislocations are line defects whose motion produces plastic deformation. An **edge dislocation** is an extra half-plane of atoms; Burgers vector b = (a,0) lies in the slip plane. A shear stress tau exerts the Peach-Koehler force tau b per unit length. In molecular statics there is no thermal activation, so the core only moves when tau exceeds the lattice (Peierls) barrier. A free surface attracts the dislocation (image force ~ mu b^2/(4 pi (1-nu) d)); when it reaches the surface it leaves and a surface step of height b remains. Lennard-Jones alone cannot hold a square lattice (pair forces give no shear stiffness, and LJ curvature is negative at the diagonal distance), so the potential has an extra Gaussian well at sqrt(2)a (Eq. 5). The input block was built from the elastic (Volterra) displacement field, which is wrong close to the core, hence it must be relaxed.
**Code.** Top two rows (`top`) and bottom two rows (`bottom`) are constrained; the rest (`mobile`) is relaxed. A Verlet list (eps = 0.2) makes it fast.
* (a) relax; locate the core by coordination number (neighbours closer than 1.2a; 4 in the perfect square lattice), ignoring atoms within 2.5a of the free surfaces and the rows next to the constrained groups. Core = atoms with coordination != 4, core position = their mean x.
* (b) loading loop: top atoms x = X + u, u = 0.02, 0.04, ... 0.5; each step starts from the previous relaxed state; the table of CG iterations and final max force per step is in `results.json -> q3 -> table`. Shear strain gamma = u/H, H = <y>_top - <y>_bottom.
* (c) reaction force R_x = -sum of f_x on top atoms *before* they are zeroed (the force the loading device must supply to hold the top group). Plots of E(u) and R_x(u).
* (d) core x versus u, Peierls stress tau = R_x/W (W = length of top group) from the last step before the core moves, and the final configuration.
**Results / explanation.**
* Relaxed block E = -4113.12 (from -4108.22), core at x about -0.28, 7 atoms of coordination 5.
* Elastic loading (R_x rises, energy rises) until the core first jumps at u = 0.10 (R_x just before: 6.79, W = 39.52, so **tau about 0.172**, in epsilon/sigma^2). After a jump the stress is partly released, then the core glides towards the right surface.
* R_x decreases and even changes sign while the core approaches the surface: the dislocation's own slip has already relaxed the shear carried by the block (each unit of core glide slips the upper half by b over the lower), and the image force pulls the core out without needing applied stress, so the top group is pushed rather than held back.
* At u = 0.32 the core reaches the surface: it leaves the detection region, the energy drops by about 17.7 (elastic energy of the dislocation released), R_x jumps to -21.8 and a step of about b remains on the right surface (`q3d_final_configuration.png`).

## Q4. Neighbour lists and minimizer efficiency  (`q4`)
**Context.** Pair loops cost O(N^2); real MD/MS codes use neighbour lists to reach O(N). The trade-off is the padding eps: a larger list is more expensive per evaluation but is rebuilt less often.
**Code.** (a) both searches are in `Evaluator` (see above); (b) relax `interstitial_{10,16,20}` with each, compare pair distances, CPU time (`time.process_time()` around the minimization only), builds, final energy; (c) eps = 0.1, 0.2, 0.35 for 20x20; (d) SD vs CG on `vacancy_20x20`, max force against number of energy/force evaluations.
**Results.**
| Cell | N2 pair distances | Verlet pair distances | N2 CPU (s) | Verlet CPU (s) | builds |
|---|---|---|---|---|---|
| 10x10 | 10.57M | 3.06M | 1.47 | 0.92 | 7 |
| 16x16 | 32.57M | 3.94M | 3.37 | 1.13 | 7 |
| 20x20 | 152.3M | 11.76M | 14.9 | 2.50 | 7 |

Energies are identical for both searches (e.g. -1153.4038461 for 20x20), as they must be. Padding on 20x20: eps = 0.1 gives list length 3634, 17 builds, 8.26M distances; eps = 0.2 gives 5879, 7, 11.76M; eps = 0.35 gives 7095, 4, 13.8M. Small eps means a short list but frequent rebuilds (each costing N(N-1)/2); large eps means few rebuilds but a longer list on each of the ~1900 evaluations. SD vs CG on the vacancy: SD 157 iterations / 771 evaluations, CG 30 / 153, same energy -1158.86598.

CPU times depend on the machine; everything else is deterministic.
