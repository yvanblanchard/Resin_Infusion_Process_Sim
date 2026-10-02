"""
RTMsim-Py test: shell part from an STL file, cascade injection, PyVista view.

Loads:
  assets/rtm_sim_test_shell.stl   CAD surface tessellation [mm]

The STL is loaded with trimesh and remeshed with MMG's surface remesher
(mmgs, through mmgpy) before simulation: CAD tessellations carry long,
flat triangles spanning whole CAD faces, which make the finite-area
gradient/flux reconstruction strongly non-orthogonal and the solver
diverge. The remeshed surface is handed to rtmsim as a trimesh object.

The part has two separate bodies (the main shell and a web touching it
along one edge, not sharing nodes), so the web gets its own port. Ports
are defined by coordinates in STL units [mm] plus a radius; cascade
ports open at their activation time.

Solver: RTMsim i_model=1 (compressible-air RTM) with the preform and
fluid values of assets/input_case3_coarsemesh.txt.

Outputs (in output_stl_shell/):
  fill_front_t50.png   PyVista view at 50 % of the total filling time:
                       cells coloured by fill time, dry cells grey, flow
                       front line, port symbols
  fill_progress.png    filled fraction vs time with port activations

Usage:
  python test_filling_frame.py
  (set OFF_SCREEN below: False = interactive PyVista window, True = file only)
"""
import os
import sys
import time
import warnings
import numpy as np
import trimesh
import mmgpy
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmsim as rtm


HERE = os.path.dirname(os.path.abspath(__file__))
STL_PATH = os.path.join(HERE, "assets", "rtm_sim_test_shell.stl")
OUTDIR = os.path.join(HERE, "output_stl_shell")
OFF_SCREEN = False        # True: render the PyVista view to file only
STL_SCALE = 1e-3          # STL units (mm) -> m
STL_UNITS = "mm"

# mmgs remeshing [mm]: target size, size bounds, Hausdorff distance to the
# input surface, size gradation.
REMESH = dict(hsiz=6.0, hmin=2.0, hausd=0.5, hgrad=1.3)

# Ports: coordinates and radius in STL units [mm], t_activate [s].
# Points are projected onto the surface; cells within `radius` of the
# projected point become inlet cells.
PORTS = [
    dict(name="P1", coords=(77697.6, 48385.1, 199853.1), radius=8.0, t_activate=0.0),
    dict(name="P2 web", coords=(77711.2, 48402.0, 199895.6), radius=6.0, t_activate=0.0),
    dict(name="C1", coords=(77686.6, 48435.1, 200027.9), radius=8.0, t_activate=80.0),
    dict(name="C2", coords=(77690.5, 48444.0, 200176.3), radius=8.0, t_activate=160.0),
]
# Vents (held at p_init) where the flow arrives last: the two corners of
# the wide top end and the top of the web. Without them the trapped air
# keeps the part from filling.
VENTS = [
    dict(name="V1", coords=(77650.3, 48483.8, 200223.0), radius=8.0),
    dict(name="V2", coords=(77754.7, 48478.9, 200216.5), radius=8.0),
    dict(name="V3 web", coords=(77701.5, 48483.2, 200220.5), radius=6.0),
]

# Process / material data (assets/input_case3_coarsemesh.txt)
TMAX = 2000.0
N_PICS = 32
P_INLET = 1.35e5
P_INIT = 1.0e5
AIR_EOS = dict(p_ref=1.01325e5, rho_ref=1.225, gamma=1.4)
MU_RESIN = 0.06
PLY = dict(thickness=3e-3, porosity=0.7, K1=3e-10, K2=3e-10, refdir=(0.0, 0.0, 1.0))

# Colours (validated categorical pair; blue sequential ramp for fill time)
COLOR_PRIMARY = "#eb6834"
COLOR_CASCADE = "#1baf7a"
COLOR_VENT = "#52514e"
COLOR_DRY = "#e1e0d9"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_SERIES = "#2a78d6"
FILL_TIME_CMAP = LinearSegmentedColormap.from_list(
    "fill_time", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                  "#256abf", "#184f95", "#0d366b"])


# ---------- mesh ----------
def load_remeshed_stl(path, remesh):
    """STL -> trimesh -> mmgs isotropic remesh -> trimesh -> ShellMesh."""
    tm = trimesh.load(path, force="mesh", process=True)
    mm = mmgpy.MmgMeshS(np.asarray(tm.vertices, dtype=np.float64),
                        np.asarray(tm.faces, dtype=np.int32))
    mm.remesh(verbose=-1, **remesh)
    tm_new = trimesh.Trimesh(vertices=mm.get_vertices(),
                             faces=mm.get_triangles(), process=True)
    mesh = rtm.ShellMesh.from_trimesh(tm_new, scale=STL_SCALE, units=STL_UNITS)
    return tm, mesh


def print_quality(label, q):
    print(f"  {label}: {q['n_cells']} cells, {q['n_nodes']} nodes, "
          f"bodies {q['body_sizes']}, area min/median/max "
          f"{q['area_min']:.3g}/{q['area_median']:.3g}/{q['area_max']:.3g} "
          f"{q['units']}^2, slivers {q['n_slivers']}")


# ---------- simulation ----------
def build_simulation(mesh):
    resin = rtm.ResinMaterial("resin").set_viscosity(MU_RESIN)
    fabric = (rtm.FabricMaterial("preform")
              .set_permeability(PLY["K1"], PLY["K2"])
              .set_porosity(PLY["porosity"]))
    stack = rtm.LaminateStack().add_ply(fabric, PLY["thickness"], PLY["refdir"])
    sim = (rtm.RTMSimulation()
           .set_mesh(mesh)
           .set_process_model(1)
           .set_resin(resin)
           .set_laminate(stack)
           .set_pressures(p_inlet=P_INLET, p_init=P_INIT)
           .set_air_eos(**AIR_EOS)
           .set_run_control(tmax=TMAX, n_pics=N_PICS)
           .set_solver_settings(h_min_mode="percentile", h_min_percentile=1.0))
    for port in PORTS:
        sim.add_injection_port(port["coords"], t_activate=port["t_activate"],
                               radius=port["radius"], name=port["name"])
    for vent in VENTS:
        sim.add_vent(vent["coords"], radius=vent["radius"], name=vent["name"])
    return sim


def print_ports(sim):
    body = sim.get_mesh().get_body_ids()
    for p in sim.get_ports():
        kind = "vent" if p.kind == "vent" else ("cascade" if p.is_cascade else "primary")
        print(f"  {p.name:7s} {kind:8s} t_act={p.t_activate:6.1f}s  "
              f"{p.cells.size:3d} cells on body {np.unique(body[p.cells]).tolist()}, "
              f"snapped {p.snap_distance:.2f} {sim.get_mesh().get_units()} "
              f"-> {np.round(p.center, 1).tolist()}")


# ---------- plotting ----------
def _add_fill_scene(pl, sim, poly, filled, t_show, show_scalar_bar):
    """Fill state at t_show, flow front and port symbols on the active subplot."""
    mesh = sim.get_mesh()
    # Mostly ambient lighting so the colour ramp reads true on the curved part.
    shading = dict(ambient=0.6, diffuse=0.4, specular=0.0)
    if (~filled).any():
        pl.add_mesh(poly.extract_cells(np.where(~filled)[0]), color=COLOR_DRY,
                    **shading)
    pl.add_mesh(poly.extract_cells(np.where(filled)[0]),
                scalars="fill time [s]", cmap=FILL_TIME_CMAP,
                clim=(0.0, t_show), n_colors=256, **shading,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args=dict(title="fill time [s]", color=COLOR_INK,
                                     vertical=True, position_x=0.86,
                                     position_y=0.25, height=0.5, width=0.06,
                                     fmt="%.0f"))
    outline = poly.extract_feature_edges(
        boundary_edges=True, feature_edges=True, feature_angle=60.0,
        manifold_edges=False, non_manifold_edges=False)
    pl.add_mesh(outline, color=COLOR_MUTED, line_width=1)

    # Flow front: mesh edges shared by a filled and a dry cell.
    tm = mesh.get_trimesh()
    adj = tm.face_adjacency
    front_edges = tm.face_adjacency_edges[filled[adj[:, 0]] != filled[adj[:, 1]]]
    if len(front_edges):
        lines = np.column_stack([np.full(len(front_edges), 2),
                                 front_edges]).ravel()
        pl.add_mesh(pv.PolyData(poly.points, lines=lines), color=COLOR_INK,
                    line_width=5, render_lines_as_tubes=True)

    r_mark = 0.012 * float(poly.length)
    labels_pts, labels_txt = [], []
    for p in sim.get_ports():
        c = np.asarray(p.center)
        if p.kind == "vent":
            glyph = pv.Cone(center=c, direction=(0.0, 0.0, 1.0),
                            height=2.2 * r_mark, radius=r_mark)
            color = COLOR_VENT
            txt = f"{p.name} (vent)"
        elif p.is_cascade:
            glyph = pv.Cube(center=c, x_length=1.6 * r_mark,
                            y_length=1.6 * r_mark, z_length=1.6 * r_mark)
            color = COLOR_CASCADE
            closed = "" if p.t_activate <= t_show else " (closed)"
            txt = f"{p.name}  t_act = {p.t_activate:.0f} s{closed}"
        else:
            glyph = pv.Sphere(radius=r_mark, center=c)
            color = COLOR_PRIMARY
            txt = f"{p.name}  t_act = 0 s"
        pl.add_mesh(glyph, color=color, smooth_shading=True)
        if p.radius > 0:
            pl.add_mesh(pv.Sphere(radius=p.radius, center=c), color=color,
                        opacity=0.18)
        port_edges = poly.extract_cells(p.cells).extract_feature_edges(
            boundary_edges=True, feature_edges=False, manifold_edges=False,
            non_manifold_edges=False)
        pl.add_mesh(port_edges, color=color, line_width=3)
        labels_pts.append(c)
        labels_txt.append(txt)
    pl.add_point_labels(np.array(labels_pts), labels_txt, font_size=13,
                        text_color=COLOR_INK, shape_color="white",
                        shape_opacity=0.8, point_size=1, always_visible=True,
                        show_points=False)


def plot_front_pyvista(sim, t_show, outpath, off_screen):
    """
    Surface at t_show, seen from both sides: cells filled by then coloured
    by fill time, dry cells grey, flow front as a black line, port symbols
    (sphere = primary, cube = cascade with activation time, cone = vent).
    """
    mesh = sim.get_mesh()
    filled = sim.get_filled_mask_at(t_show)
    fill_time = np.where(filled, sim.get_fill_time_field(), 0.0)
    poly = mesh.to_pyvista(cell_data={"fill time [s]": fill_time})

    pl = pv.Plotter(shape=(1, 2), off_screen=off_screen,
                    window_size=(1800, 1000), border=False)
    pl.set_background("#fcfcfb")
    views = [("front side", 200.0), ("back side", 20.0)]
    for k, (label, azimuth) in enumerate(views):
        pl.subplot(0, k)
        _add_fill_scene(pl, sim, poly, filled, t_show,
                        show_scalar_bar=(k == len(views) - 1))
        pl.add_text(label, position="lower_edge", font_size=11,
                    color=COLOR_MUTED)
        pl.view_isometric()
        pl.camera.azimuth = azimuth
        pl.camera.elevation = 10
        pl.reset_camera()
        pl.camera.zoom(1.1)

    pl.subplot(0, 0)
    t_fill = sim.get_total_fill_time()
    pct = filled[sim.get_fill_region()].mean() * 100
    pl.add_text(f"Flow front at 50 % of filling time: t = {t_show:.0f} s of "
                f"{t_fill:.0f} s ({pct:.0f} % of fed cells filled)\n"
                f"black line = flow front, grey = dry; sphere = primary port, "
                f"cube = cascade port, cone = vent",
                position="upper_left", font_size=11, color=COLOR_INK)
    pl.add_axes(color=COLOR_INK)
    if off_screen:
        pl.screenshot(outpath)
        pl.close()
    else:
        pl.show(screenshot=outpath)


def plot_fill_curve(sim, t_show, outpath):
    """Filled fraction of the fed cells vs time, from the fill-time field."""
    cells = (sim.get_fill_region()
             & (sim.get_final_snapshot().celltype != rtm.CELL_OUTLET))
    t_fill = sim.get_total_fill_time()
    times = np.linspace(0.0, t_fill, 500)
    frac = [np.mean(sim.get_filled_mask_at(t)[cells]) for t in times]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(times, frac, "-", color=COLOR_SERIES, lw=2)
    for p in sim.get_ports():
        if p.is_cascade:
            ax.axvline(p.t_activate, color=COLOR_MUTED, ls="--", lw=1)
            ax.annotate(f"{p.name} opens", (p.t_activate, 0.02),
                        rotation=90, va="bottom", ha="right",
                        fontsize=8, color="#52514e")
    ax.axvline(t_show, color=COLOR_INK, ls=":", lw=1)
    ax.annotate("50 % fill time", (t_show, 0.98), rotation=90, va="top",
                ha="right", fontsize=8, color="#52514e")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("filled fraction of fed cells (γ ≥ 0.5)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("STL shell — fill progress")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"Reading STL: {STL_PATH}")
    t0 = time.time()
    tm_in, mesh = load_remeshed_stl(STL_PATH, REMESH)
    print_quality("input STL", rtm.ShellMesh.from_trimesh(
        tm_in, scale=STL_SCALE, units=STL_UNITS).get_quality_report())
    print_quality(f"mmgs remesh {REMESH}", mesh.get_quality_report())
    print(f"  -> {time.time() - t0:.1f}s")

    sim = build_simulation(mesh)
    print("\nPorts:")
    print_ports(sim)

    print("\nRunning solver (first call may JIT-compile)...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        snaps = sim.run()
    print(f"  -> {time.time() - t0:.1f}s, {len(snaps)} snapshots")

    t_fill = sim.get_total_fill_time()
    if not sim.is_fill_complete():
        print(f"WARNING: fill criterion not reached by tmax={TMAX:.0f} s; "
              f"using the end time {t_fill:.1f} s")
    t_show = 0.5 * t_fill
    final = sim.get_final_snapshot()
    region = sim.get_fill_region()
    print(f"\nTotal filling time = {t_fill:.1f} s "
          f"(final mean gamma {final.gamma[final.get_fluid_mask() & region].mean():.3f})")
    print(f"Front shown at 50 % of filling time: t = {t_show:.1f} s, "
          f"{sim.get_filled_mask_at(t_show)[region].mean()*100:.1f} % of fed cells filled")

    plot_fill_curve(sim, t_show, os.path.join(OUTDIR, "fill_progress.png"))
    plot_front_pyvista(sim, t_show, os.path.join(OUTDIR, "fill_front_t50.png"),
                       off_screen=OFF_SCREEN)
    print(f"\nFigures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
