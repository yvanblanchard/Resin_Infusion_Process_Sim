"""
RTMsim-Py: verification of the incompressible model (i_model=4) against
exact solutions and conservation checks.

  V1  1-D channel: front x = sqrt(2 K dp t / (phi mu))            < 1 %
  V2a flat plate, radial flow from the 50 mm port of the double dome
      case: fill time vs the exact radial solution, < 2 % at 10 mm. The
      exact solution uses the port as meshed (cells with centres inside
      25 mm, equivalent radius from their area, which differs per mesh);
      errors against the nominal 25 mm port are printed too.
  V2b mesh convergence at 20/10/5 mm with a port that is the same on
      every mesh (40 mm square on cell boundaries; exact far field: radial
      flow from its conformal radius 0.5902 x side): error decreasing at
      every radius, Richardson-extrapolated error < 0.5 % from 6 conformal
      radii outwards (closer in, the far-field reference is approximate)
  V3  anisotropic radial flow, K1/K2 = 5 at 30 deg, from an elliptical
      port of the same anisotropy (the coordinate change x' = x
      sqrt(K/K1), y' = y sqrt(K/K2) makes it plain radial flow, so the
      front is an exact ellipse): axis ratio sqrt(5), angle 30 deg,
      filled area vs the exact radial solution
  V4  anisotropic strip bent into a half cylinder vs the same strip flat:
      identical fill times (checks the element frames)
  V5  resin volume balance (injected = stored + vented)          < 1e-10
  V6  plate filled to completion: every cell full, vents included,
      is_fill_complete() True
  V7  trapped air: centre inlet, one corner vent, RTM at 1 bar. Dry spots
      in the three other corners only, with the air of a corner at the
      moment the front reaches the edges; pocket pressure in balance with
      the resin around it. The same at 300 Pa (pockets compressed ~300x)
      and in vacuum (fills completely)

Run:  python verification_model4.py [--quick]      (--quick skips 5 mm)
Outputs: output_verification_model4/v2_radial_convergence.png,
         v3_anisotropic_front.png
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # repo root (rtmsim.py)
import rtmsim as rtm

# Flat-plate case of the double dome validation (Pierce & Falzon 2017)
K = 3.3e-11
PHI = 0.724
MU = 0.0756
H = 0.4e-3
P_IN = 101.3e3
P_OUT = 0.3e3
R0 = 0.025

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MESH_COLORS = {20.0: "#5598e7", 10.0: "#256abf", 5.0: "#104281"}  # blue ramp
MESH_MARKERS = {20.0: "o", 10.0: "s", 5.0: "D"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9, "legend.frameon": False,
})


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def grid_mesh(nx, ny, mapping):
    """Structured triangle mesh of [0, 1]^2 mapped by mapping(u, v) -> xyz."""
    u, v = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1),
                       indexing="ij")
    nodes = mapping(u.ravel(), v.ravel())
    nid = lambda i, j: i * (ny + 1) + j
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += ([[a, b, c], [a, c, d]] if (i + j) % 2 == 0
                     else [[a, b, d], [b, c, d]])
    return rtm.ShellMesh(nodes, np.array(tris))


def plate(side, spacing):
    n_div = int(round(side / spacing))
    mesh = rtm.ShellMesh.make_square_plate(side, n_div)
    cc = mesh.cellcenter
    ring = np.where(np.max(np.abs(cc[:, :2]), axis=1)
                    > side / 2 - side / n_div)[0]
    return mesh, ring


def make_sim(mesh, fabric_K, tmax, n_pics=8, p_in=P_IN, p_init=P_OUT,
             refdir=(1, 0, 0), angle_deg=0.0):
    fab = (rtm.FabricMaterial("fabric").set_permeability(*fabric_K)
           .set_porosity(PHI))
    return (rtm.RTMSimulation().set_mesh(mesh).set_process_model(4)
            .set_resin(rtm.ResinMaterial("oil").set_viscosity(MU))
            .set_laminate(rtm.LaminateStack().add_ply(
                fab, H, refdir=refdir, angle_deg=angle_deg))
            .set_pressures(p_inlet=p_in, p_init=p_init)
            .set_run_control(tmax=tmax, n_pics=n_pics))


def run(sim, label):
    t0 = time.time()
    sim.run()
    st = sim.get_run_stats()
    print(f"    {label}: {time.time() - t0:.1f} s wall, {st['n_steps']} steps, "
          f"{st['n_solves']} solves, "
          f"{'complete' if sim.is_fill_complete() else 'not complete'}")
    return sim


def radial_time(r, r0, k=K, dp=P_IN - P_OUT):
    """Exact fill time of radial Darcy flow from a port of radius r0."""
    c = PHI * MU / (k * dp)
    return c * (r ** 2 / 2 * np.log(r / r0) - (r ** 2 - r0 ** 2) / 4)


def radial_front(t, r0, k=K):
    return brentq(lambda r: radial_time(r, r0, k) - t, r0 * (1 + 1e-9), 10.0)


def port_cells(sim):
    return np.concatenate([p.cells for p in sim.get_ports()
                           if p.kind == "inlet"])


RESULTS = []


def check(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")


# --------------------------------------------------------------------------
# V1: 1-D channel
# --------------------------------------------------------------------------
def v1_channel():
    print("\n[V1] 1-D channel")
    L, W, nx = 0.5, 0.02, 100
    mesh = grid_mesh(nx, 4, lambda u, v: np.column_stack(
        [L * u, W * v, np.zeros_like(u)]))
    cc = mesh.cellcenter
    dx = L / nx
    sim = make_sim(mesh, (K, K), tmax=1e4, n_pics=200)
    sim.add_injection_port_cells(np.where(cc[:, 0] < dx)[0])
    sim.add_vent_cells(np.where(cc[:, 0] > L - dx)[0])
    run(sim, f"{mesh.N} cells")
    area = mesh.get_cell_areas()
    errs = []
    for s in sim.get_snapshots()[1:]:
        # Front from the resin volume beyond the port boundary x = dx.
        x_f = (s.gamma * area)[s.celltype != rtm.CELL_INLET].sum() / W
        if 0.05 < x_f < L - 2 * dx:
            x_ex = np.sqrt(2 * K * (P_IN - P_OUT) * s.t / (PHI * MU))
            errs.append(x_f / x_ex - 1)
    e = float(np.max(np.abs(errs)))
    check("V1 1-D front", e < 0.01, f"max |front error| {100 * e:.3f} % "
          f"over {len(errs)} snapshots")
    return sim


# --------------------------------------------------------------------------
# V2: radial flow, mesh convergence
# --------------------------------------------------------------------------
def radial_errors(sim, mesh, r0, spacing, radii):
    """Per-cell fill-time error vs radial flow from r0 at the cell's own
    radius, median over a ring of one cell width around each radius (a
    ring median of fill times alone is biased: t grows ~ r^2 across it)."""
    ft = sim.get_fill_time_field()
    r = np.linalg.norm(mesh.cellcenter[:, :2], axis=1)
    ok = np.isfinite(ft) & (r > 1.5 * r0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(ok, ft / radial_time(np.maximum(r, r0), r0) - 1, np.nan)
    return np.array([np.nanmedian(e[np.abs(r - b) < spacing / 2])
                     for b in radii])


def v2_radial(outdir, spacings):
    print("\n[V2a] Flat plate, radial flow from the 50 mm circular port")
    radii = np.array([0.088, 0.15, 0.21, 0.28, 0.32])
    r_txt = ", ".join(f"{1e3 * b:.0f}" for b in radii)
    max_a = {}
    for sp in spacings:
        mesh, vent = plate(0.8, sp * 1e-3)
        sim = make_sim(mesh, (K, K), tmax=1800.0)
        sim.add_injection_port((0, 0, 0), radius=R0).add_vent_cells(vent)
        run(sim, f"{sp:g} mm, {mesh.N} cells")
        area = mesh.get_cell_areas()
        r0_mesh = np.sqrt(area[port_cells(sim)].sum() / np.pi)
        e_mesh = radial_errors(sim, mesh, r0_mesh, sp * 1e-3, radii)
        e_nom = radial_errors(sim, mesh, R0, sp * 1e-3, radii)
        max_a[sp] = float(np.max(np.abs(e_mesh)))
        print(f"      meshed port: cells with centres inside 25 mm, "
              f"equivalent radius {1e3 * r0_mesh:.1f} mm")
        print(f"      fill-time error at r = {r_txt} mm")
        print("        vs meshed port:  "
              + "  ".join(f"{100 * x:+5.1f} %" for x in e_mesh))
        print("        vs 25 mm port:   "
              + "  ".join(f"{100 * x:+5.1f} %" for x in e_nom))
    if 10.0 in max_a:
        check("V2a radial, 10 mm", max_a[10.0] < 0.02,
              f"max |fill time error| {100 * max_a[10.0]:.2f} % (vs meshed "
              f"port)")

    # Convergence needs the same port on every mesh: a 40 mm square on
    # cell boundaries. Far from it the flow is radial from the square's
    # conformal radius 0.5902 x side.
    print("\n[V2b] Mesh convergence, 40 mm square port (same on every mesh)")
    side = 0.04
    r0_sq = 0.59017 * side
    rb_plot = np.arange(0.07, 0.33, 0.01)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    errs = {}
    for sp in spacings:
        mesh, vent = plate(0.8, sp * 1e-3)
        port = np.where(np.max(np.abs(mesh.cellcenter[:, :2]), axis=1)
                        < side / 2)[0]
        sim = make_sim(mesh, (K, K), tmax=1800.0)
        sim.add_injection_port_cells(port).add_vent_cells(vent)
        run(sim, f"{sp:g} mm, {mesh.N} cells")
        errs[sp] = radial_errors(sim, mesh, r0_sq, sp * 1e-3, radii)
        print(f"      fill-time error at r = {r_txt} mm:  "
              + "  ".join(f"{100 * x:+5.2f} %" for x in errs[sp]))
        ep = radial_errors(sim, mesh, r0_sq, sp * 1e-3, rb_plot)
        ax.plot(rb_plot * 1e3, 100 * ep, marker=MESH_MARKERS[sp], ms=4,
                lw=1.5, color=MESH_COLORS[sp], mec=SURFACE, mew=0.6,
                label=f"{sp:g} mm cells")
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.set_xlabel("radius [mm]")
    ax.set_ylabel("fill time error [%]")
    ax.set_title("Radial flow from a 40 mm square port: error vs the exact "
                 "solution", fontsize=9, color=INK2, loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "v2_radial_convergence.png"), dpi=150)
    plt.close(fig)
    sps = sorted(errs, reverse=True)
    conv = len(sps) > 1 and all(np.all(np.abs(errs[a]) > np.abs(errs[b]))
                                for a, b in zip(sps[:-1], sps[1:]))
    detail = ", ".join(f"{s:g} mm: {100 * np.abs(errs[s]).max():.2f} %"
                       for s in sps)
    if len(sps) >= 3:
        # Richardson extrapolation at each radius from the three finest.
        # From r >= 6 conformal radii only: closer in, the far-field
        # reference itself is approximate (square-port near field).
        e1, e2, e3 = (errs[s] for s in sps[-3:])
        order = np.log2((e1 - e2) / (e2 - e3))
        e_inf = e3 - (e2 - e3) / (2.0 ** order - 1.0)
        far = radii >= 6.0 * r0_sq
        detail += (f"; observed order {order.min():.1f}..{order.max():.1f}, "
                   f"extrapolated error {100 * np.abs(e_inf[far]).max():.2f} %"
                   f" at r >= {1e3 * radii[far].min():.0f} mm "
                   f"({100 * e_inf[~far].max():.2f} % closer in)")
        conv = conv and np.abs(e_inf[far]).max() < 0.005
    check("V2b convergence", conv, detail)


# --------------------------------------------------------------------------
# V3: anisotropic radial flow
# --------------------------------------------------------------------------
def filled_ellipse(gamma, area, xy):
    """Axis ratio, major-axis angle [deg] and area of the filled region,
    from its second moments (exact for a uniformly filled ellipse)."""
    w = gamma * area
    A = w.sum()
    d = xy - (w[:, None] * xy).sum(0) / A
    C = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / A
    lam, vec = np.linalg.eigh(C)
    return (np.sqrt(lam[1] / lam[0]),
            np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180.0, A)


def v3_anisotropic(outdir, spacings):
    print("\n[V3] Anisotropic radial flow, K1/K2 = 5 at 30 deg")
    ratio, ang, r0 = 5.0, 30.0, 0.035
    K1, K2 = K * np.sqrt(ratio), K / np.sqrt(ratio)
    th = np.radians(ang)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8),
                             gridspec_kw=dict(width_ratios=(1.0, 1.25)))
    res = {}
    for sp in spacings:
        mesh, vent = plate(0.8, sp * 1e-3)
        cc = mesh.cellcenter
        xr = cc[:, 0] * np.cos(th) + cc[:, 1] * np.sin(th)
        yr = -cc[:, 0] * np.sin(th) + cc[:, 1] * np.cos(th)
        port = np.where((xr / (r0 * ratio ** 0.25)) ** 2
                        + (yr / (r0 * ratio ** -0.25)) ** 2 <= 1.0)[0]
        sim = make_sim(mesh, (K1, K2), tmax=800.0, n_pics=8, angle_deg=ang)
        sim.add_injection_port_cells(port).add_vent_cells(vent)
        run(sim, f"{sp:g} mm, {mesh.N} cells")
        area = mesh.get_cell_areas()
        r0_mesh = np.sqrt(area[port].sum() / np.pi)   # area-preserving map
        rows = []
        for s in sim.get_snapshots()[1:]:
            ar, an, A = filled_ellipse(s.gamma, area, cc[:, :2])
            rows.append((s.t, ar, an,
                         np.sqrt(A / np.pi) / radial_front(s.t, r0_mesh) - 1))
        rows = np.array(rows)
        res[sp] = rows
        t, ar, an, ea = rows[-1]
        print(f"      t = {t:.0f} s: axis ratio {ar:.3f} (exact "
              f"{np.sqrt(ratio):.3f}, {100 * (ar / np.sqrt(ratio) - 1):+.1f} %),"
              f" angle {an:.1f} deg, equivalent radius {100 * ea:+.2f} %")
        axes[1].plot(rows[:, 0], rows[:, 1], marker=MESH_MARKERS[sp], ms=4,
                     lw=1.5, color=MESH_COLORS[sp], mec=SURFACE, mew=0.6,
                     label=f"{sp:g} mm cells")
        if sp == min(spacings):
            snap = sim.get_final_snapshot()
            tri = mesh.faces
            axes[0].tripcolor(mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3,
                              tri, facecolors=snap.gamma, cmap="Blues",
                              vmin=0, vmax=1.6, edgecolors="none")
            # Exact front: ellipse of the transformed radial solution.
            rf = radial_front(snap.t, r0_mesh)
            a_, b_ = rf * ratio ** 0.25, rf * ratio ** -0.25
            ph = np.linspace(0, 2 * np.pi, 200)
            ex = a_ * np.cos(ph) * np.cos(th) - b_ * np.sin(ph) * np.sin(th)
            ey = a_ * np.cos(ph) * np.sin(th) + b_ * np.sin(ph) * np.cos(th)
            axes[0].plot(ex * 1e3, ey * 1e3, color=ORANGE, lw=1.5,
                         label="exact front")
            axes[0].set_aspect("equal")
            axes[0].set_xlim(-400, 400)
            axes[0].set_ylim(-400, 400)
            axes[0].grid(False)
            axes[0].set_xlabel("x [mm]")
            axes[0].set_ylabel("y [mm]")
            axes[0].set_title(f"Filled region at t = {snap.t:.0f} s, "
                              f"{sp:g} mm cells", fontsize=9, color=INK2,
                              loc="left")
            axes[0].legend(loc="upper left")
    axes[1].axhline(np.sqrt(ratio), color=ORANGE, lw=1.2,
                    label="exact, √5")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("front axis ratio")
    axes[1].set_title("Axis ratio of the filled region", fontsize=9,
                      color=INK2, loc="left")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "v3_anisotropic_front.png"), dpi=150)
    plt.close(fig)

    sp = min(spacings)
    rows = res[sp]
    e_ar = np.abs(rows[:, 1] / np.sqrt(ratio) - 1)
    e_an = np.abs(rows[:, 2] - ang)
    e_A = np.abs(rows[:, 3])
    check(f"V3 axis ratio ({sp:g} mm)", e_ar[-1] < 0.03,
          f"{rows[-1, 1]:.3f} vs {np.sqrt(ratio):.3f} at t = {rows[-1, 0]:.0f} s"
          f" ({100 * e_ar[-1]:.1f} %)")
    check("V3 angle", e_an.max() < 1.0,
          f"max |angle - 30 deg| {e_an.max():.2f} deg")
    check("V3 timing", e_A.max() < 0.02,
          f"max |equivalent radius error| {100 * e_A.max():.2f} %")
    if len(res) > 1:
        sps = sorted(res, reverse=True)
        errs = [abs(res[s][-1, 1] / np.sqrt(ratio) - 1) for s in sps]
        check("V3 axis ratio converges",
              all(a > b for a, b in zip(errs[:-1], errs[1:])),
              ", ".join(f"{s:g} mm: {100 * e:.1f} %" for s, e in zip(sps, errs)))


# --------------------------------------------------------------------------
# V4: curved strip vs flat strip
# --------------------------------------------------------------------------
def v4_curved():
    print("\n[V4] Half-cylinder strip vs flat strip (K1/K2 = 5 at 30 deg)")
    R, W = 0.1, 0.05
    S = np.pi * R
    nx, ny = 63, 10
    flat = grid_mesh(nx, ny, lambda u, v: np.column_stack(
        [S * u, W * v, np.zeros_like(u)]))
    bent = grid_mesh(nx, ny, lambda u, v: np.column_stack(
        [R * np.sin(S * u / R), W * v, R * np.cos(S * u / R)]))
    s_c = flat.cellcenter[:, 0]       # same cell order on both meshes
    inlet = np.where(s_c < S / nx)[0]
    vent = np.where(s_c > S - S / nx)[0]
    th = s_c / R
    tangents = {"flat": np.tile([1.0, 0.0, 0.0], (flat.N, 1)),
                "bent": np.column_stack([np.cos(th), np.zeros_like(th),
                                         -np.sin(th)])}
    ft = {}
    for name, mesh in (("flat", flat), ("bent", bent)):
        sim = make_sim(mesh, (K * np.sqrt(5), K / np.sqrt(5)), tmax=1e5,
                       refdir=tangents[name], angle_deg=30.0)
        sim.add_injection_port_cells(inlet).add_vent_cells(vent)
        run(sim, f"{name}, {mesh.N} cells")
        ft[name] = sim.get_fill_time_field()
    sel = ft["flat"] > 0
    d = np.abs(ft["bent"][sel] / ft["flat"][sel] - 1)
    check("V4 curved = flat", d.max() < 2e-3,
          f"max relative fill-time difference {d.max():.1e} "
          f"(flat fill {np.nanmax(ft['flat']):.0f} s)")


# --------------------------------------------------------------------------
# V5 / V6: full fill and volume balance
# --------------------------------------------------------------------------
def v6_full_fill():
    print("\n[V6] Plate filled to completion")
    mesh, vent = plate(0.4, 0.01)
    sim = make_sim(mesh, (K, K), tmax=1e5)
    sim.add_injection_port((0, 0, 0), radius=R0).add_vent_cells(vent)
    run(sim, f"{mesh.N} cells")
    g = sim.get_final_snapshot().gamma
    check("V6 complete", sim.is_fill_complete() and np.all(g == 1.0)
          and np.all(g[vent] == 1.0),
          f"complete={sim.is_fill_complete()}, cells not full: "
          f"{int((g < 1).sum())} (vents: {int((g[vent] < 1).sum())}), "
          f"t_fill = {sim.get_total_fill_time():.0f} s")
    return sim


def v5_balance(sims):
    print("\n[V5] Resin volume balance")
    worst = 0.0
    for label, sim in sims:
        b = sim.get_volume_balance()
        worst = max(worst, b["relative_error"])
        print(f"      {label}: injected {b['injected']:.4e} m^3, stored "
              f"{b['stored']:.4e}, vented {b['vented']:.4e}, spilled "
              f"{b['spilled']:.1e}, relative error {b['relative_error']:.1e}")
    check("V5 volume balance", worst < 1e-10,
          f"worst relative error {worst:.1e} over {len(sims)} runs")


# --------------------------------------------------------------------------
# V7: trapped air
# --------------------------------------------------------------------------
def v7_trapped_air():
    print("\n[V7] Trapped air: centre inlet, vent in the (+x, +y) corner")
    side, sp = 0.3, 0.005
    mesh = rtm.ShellMesh.make_square_plate(side, int(round(side / sp)))
    cc = mesh.cellcenter
    # Air of one corner when the radial front touches the four edges.
    a = side / 2
    v0_corner = (a * a - np.pi * a * a / 4) * H * PHI
    sims = []
    for p_init, p_in in ((1e5, 2e5), (300.0, P_IN), (0.0, P_IN)):
        sim = make_sim(mesh, (K, K), tmax=1e5, p_in=p_in, p_init=p_init)
        sim.add_injection_port((0, 0, 0), radius=0.02)
        sim.add_vent((a, a, 0), radius=0.02)
        run(sim, f"p_init = {p_init:g} Pa, p_inlet = {p_in:g} Pa")
        sims.append((f"V7 p_init = {p_init:g} Pa", sim))
        spots = sim.get_dry_spots()
        snap = sim.get_final_snapshot()
        corners = []
        for d in spots:
            c = cc[d["cells"]].mean(0)
            corners.append((int(np.sign(c[0])), int(np.sign(c[1]))))
            # Resin pressure around the pocket: full cells sharing an edge.
            ring = np.setdiff1d(np.unique(
                mesh.get_trimesh().face_adjacency[np.isin(
                    mesh.get_trimesh().face_adjacency, d["cells"]).any(1)]),
                d["cells"])
            d["p_ring"] = float(snap.p[ring].mean())
            print(f"      dry spot at ({1e3 * c[0]:+.0f}, {1e3 * c[1]:+.0f}) mm: "
                  f"{d['cells'].size} cells, p = {d['pressure']:.0f} Pa "
                  f"(resin around it {d['p_ring']:.0f} Pa), air "
                  f"{d['air_volume']:.3e} m^3, at p_init "
                  f"{d['air_volume_at_p_init']:.3e} m^3")
        if p_init == 0.0:
            check("V7 vacuum fills", sim.is_fill_complete() and not spots,
                  f"complete={sim.is_fill_complete()}, {len(spots)} dry spots")
            continue
        expected = {(-1, -1), (1, -1), (-1, 1)}
        check(f"V7 dry spots, p_init = {p_init:g} Pa",
              len(spots) == 3 and set(corners) == expected,
              f"{len(spots)} dry spots in corners {sorted(corners)}")
        v_trap = np.array([d["air_volume_at_p_init"] for d in spots])
        e_v = np.abs(v_trap / v0_corner - 1)
        check(f"V7 trapped air, p_init = {p_init:g} Pa", e_v.max() < 0.05,
              f"air at p_init {v_trap.min():.3e}..{v_trap.max():.3e} m^3 vs "
              f"corner beyond a circular front {v0_corner:.3e} "
              f"(max {100 * e_v.max():.1f} %)")
        e_p = max(abs(d["pressure"] - d["p_ring"]) for d in spots) / (p_in - p_init)
        ratio = min(d["pressure"] / p_init for d in spots)
        check(f"V7 pocket pressure, p_init = {p_init:g} Pa",
              e_p < 0.02 and all(p_init < d["pressure"] < p_in for d in spots),
              f"|p_pocket - p_resin around| <= {100 * e_p:.2f} % of dp; "
              f"compression p/p_init >= {ratio:.2f}")
    return sims


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true",
                    help="skip the 5 mm meshes of V2 / V3")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    outdir = os.path.join(HERE, "output_verification_model4")
    os.makedirs(outdir, exist_ok=True)
    spacings = (20.0, 10.0) if args.quick else (20.0, 10.0, 5.0)

    sims = [("V1 channel", v1_channel())]
    v2_radial(outdir, spacings)
    v3_anisotropic(outdir, spacings[1:])
    v4_curved()
    sims.append(("V6 plate", v6_full_fill()))
    sims += v7_trapped_air()
    v5_balance(sims)

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed; "
          f"figures in {outdir}")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
