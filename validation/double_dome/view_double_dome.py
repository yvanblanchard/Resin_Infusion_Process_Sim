"""
RTMsim-Py: 3-D PyVista viewer for the double dome validation case
(validation_double_dome.py, Pierce & Falzon 2017).

Runs one case (sample orientation x permeability model) on the draped ply
and shows the full part (quarter model mirrored about both symmetry
planes) on the tool, in two linked views:

  left   filling state at the slider time: filled cells coloured by fill
         time, dry cells light grey, simulated flow front (black line),
         measured flow front of the nearest experimental time (orange line,
         draped onto the tool from the paper's plan-view data), inlet
         (orange) and vent cells (dark grey)
  right  draped ply coloured by shear angle, with the K1 principal
         permeability direction of each element (Eq. 11 of the paper)

Keys 1-4 jump to the four measured times of the case.

Run:
  python view_double_dome.py                       # 0/90, shear-dependent K
  python view_double_dome.py --case 45 --variant iso
  python view_double_dome.py --off-screen --time 1255   # PNG only
  python view_double_dome.py --tmax 4000           # run on to the end of fill
Options: --geometry stl|parametric (default stl), --spacing 10 [mm].

The default run stops just after the last measured time (1795 / 1340 s),
before the far ends of the ply are reached.
"""
import argparse
import os
import sys

import numpy as np
import pyvista as pv
import trimesh
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))   # repo root (rtmsim.py)
sys.path.insert(0, HERE)                             # validation_double_dome.py
import rtmsim as rtm
import validation_double_dome as vdd


# Colours (same roles as test_filling_frame.py / validation_double_dome.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
DRY = "#e1e0d9"
TOOL = "#c3c2b7"
EXP = "#eb6834"         # measured front, inlet
VENT = "#52514e"
FILL_TIME_CMAP = LinearSegmentedColormap.from_list(
    "fill_time", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                  "#256abf", "#184f95", "#0d366b"])
SHADING = dict(ambient=0.55, diffuse=0.45, specular=0.0)

PLY_OFFSET = 1.5        # [mm] ply drawn this far above the tool (no z-fight)
LINE_OFFSET = 3.0       # [mm] fronts drawn above the ply
MIRRORS = ((1, 1), (-1, 1), (1, -1), (-1, -1))


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def mirror_quarter(points, tris):
    """Full part from the quarter: 4 mirrored copies, shared nodes merged.
    Returns points, triangles and the quarter cell id of every full cell."""
    pts, faces, src = [], [], []
    n = len(points)
    for k, (sx, sy) in enumerate(MIRRORS):
        pts.append(points * np.array([sx, sy, 1.0]))
        t = tris + k * n
        faces.append(t[:, ::-1] if sx * sy < 0 else t)  # keep normals up
        src.append(np.arange(len(tris)))
    pts = np.vstack(pts)
    faces = np.vstack(faces)
    key = np.round(pts, 6)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    return pts[first], inv.ravel()[faces], np.concatenate(src)


def mirror_vectors(vec):
    return np.vstack([vec * np.array([sx, sy, 1.0]) for sx, sy in MIRRORS])


def lift(points, offset):
    """Move points `offset` mm off the tool along its upward normal."""
    _, hx, hy = vdd.tool_height(points[:, 0], points[:, 1])
    nrm = np.column_stack([-hx, -hy, np.ones_like(hx)])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return points + offset * nrm


def tool_surface(extent=(470.0, 270.0), step=4.0):
    """Tool surface as a structured grid sampled from the height field."""
    x = np.arange(-extent[1], extent[1] + 1e-9, step)
    y = np.arange(-extent[0], extent[0] + 1e-9, step)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z, _, _ = vdd.tool_height(X, Y)
    return pv.StructuredGrid(X, Y, Z)


def drape_plan_curve(curve, step=2.0):
    """Paper's plan-view front polyline -> 3-D lines on the tool (4 copies)."""
    c = np.asarray(curve, dtype=float)
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    si = np.linspace(0.0, s[-1], max(int(s[-1] / step), 2))
    xy = np.column_stack([np.interp(si, s, c[:, 0]),
                          np.interp(si, s, c[:, 1])])
    z, _, _ = vdd.tool_height(xy[:, 0], xy[:, 1])
    base = lift(np.column_stack([xy, z]), LINE_OFFSET)
    lines = []
    for sx, sy in MIRRORS:
        p = base * np.array([sx, sy, 1.0])
        lines.append(pv.lines_from_points(p))
    return lines[0].merge(lines[1:])


# --------------------------------------------------------------------------
# Case
# --------------------------------------------------------------------------
class Case:
    def __init__(self, case_key, variant, spacing, tmax=None):
        self.key = case_key
        self.case = vdd.CASES[case_key]
        self.tmax = tmax or self.case["tmax"]
        self.variant = variant
        self.paper = vdd.PAPER[case_key]
        self.drape = dr = vdd.drape_ply(self.case["fibre_deg"], spacing)
        mesh = rtm.ShellMesh.from_arrays(dr["xyz"], dr["tris"], scale=1e-3,
                                         units="mm")
        sim = vdd.build_sim(mesh, dr, self.case["mu"], self.tmax,
                            2, variant == "shear")
        print(f"Running {self.case['label']}, {variant} K, {mesh.N} cells "
              f"(quarter)...")
        self.fill_time_q = vdd.run_sim(sim, variant)
        self.vent_cells_q = np.concatenate(
            [p.cells for p in sim.get_ports() if p.kind == "vent"])
        self.inlet_cells_q = np.concatenate(
            [p.cells for p in sim.get_ports() if p.kind == "inlet"])

        pts, faces, self.src = mirror_quarter(dr["xyz"], dr["tris"])
        cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
        self.poly = pv.PolyData(pts, cells)
        self.poly.cell_data["fill time [s]"] = self.fill_time_q[self.src]
        self.poly.cell_data["shear angle [deg]"] = np.abs(dr["shear"])[self.src]
        self.poly.points = lift(self.poly.points, PLY_OFFSET)
        self.tm = trimesh.Trimesh(self.poly.points, faces, process=False)
        # Vent cells are pinned at the outlet pressure by the solver and
        # never fill: drawn as their own category, not counted as dry ply.
        self.vent = np.isin(self.src, self.vent_cells_q)

        e1 = mirror_vectors(vdd.k1_direction(dr))
        centres = lift(self.poly.cell_centers().points, 2.0)
        half = 0.4 * spacing
        seg = np.empty((2 * len(centres), 3))
        seg[0::2] = centres - half * e1
        seg[1::2] = centres + half * e1
        lines = np.column_stack([np.full(len(centres), 2),
                                 np.arange(0, 2 * len(centres), 2),
                                 np.arange(1, 2 * len(centres), 2)]).ravel()
        self.k1_lines = pv.PolyData(seg, lines=lines)

    def filled_at(self, t):
        ft = self.poly.cell_data["fill time [s]"]
        return np.isfinite(ft) & (ft <= t) & ~self.vent

    def front_lines(self, filled):
        adj = self.tm.face_adjacency
        a, b = adj[:, 0], adj[:, 1]
        keep = (filled[a] != filled[b]) & ~self.vent[a] & ~self.vent[b]
        edges = self.tm.face_adjacency_edges[keep]
        if not len(edges):
            return None
        pts = self.poly.points
        lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
        return pv.PolyData(pts + np.array([0, 0, LINE_OFFSET - PLY_OFFSET]),
                           lines=lines)

    def port_meshes(self):
        ids_full = lambda q: np.where(np.isin(self.src, q))[0]
        inlet = self.poly.extract_cells(ids_full(self.inlet_cells_q))
        vent = self.poly.extract_cells(ids_full(self.vent_cells_q))
        return inlet, vent


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
def _bar_face(height):
    """Legend glyph: a horizontal bar (thin = line entry, thick = area)."""
    h = 0.5 * height
    return pv.PolyData(np.array([[-0.5, -h, 0], [0.5, -h, 0],
                                 [0.5, h, 0], [-0.5, h, 0]], dtype=float),
                       [4, 0, 1, 2, 3])


LINE_FACE = _bar_face(0.18)
AREA_FACE = _bar_face(0.8)
DOT_FACE = pv.Polygon(radius=0.4, n_sides=24)


def add_legend_box(pl, entries, position, size):
    """
    Legend in the active subplot: coloured glyphs, labels in primary ink
    (pyvista's add_legend writes each label in its glyph colour, which makes
    the light-grey entries unreadable). position/size: normalised viewport.
    """
    import vtk
    leg = vtk.vtkLegendBoxActor()
    leg.SetNumberOfEntries(len(entries))
    ink = pv.Color(INK).float_rgb
    for i, (label, color, face) in enumerate(entries):
        sym = face.copy()
        rgb = np.tile(np.array(pv.Color(color).int_rgb, dtype=np.uint8),
                      (sym.n_cells, 1))
        sym.cell_data.set_array(rgb, "rgb")
        sym.cell_data.active_scalars_name = "rgb"
        leg.SetEntry(i, sym, label, ink)
    leg.ScalarVisibilityOn()
    leg.UseBackgroundOn()
    leg.SetBackgroundColor(pv.Color(SURFACE).float_rgb)
    leg.SetBackgroundOpacity(0.9)
    leg.BorderOff()
    tp = leg.GetEntryTextProperty()
    tp.SetColor(ink)
    tp.SetFontFamilyToArial()
    tp.BoldOff()
    tp.ShadowOff()
    leg.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    leg.GetPositionCoordinate().SetValue(*position)
    leg.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
    leg.GetPosition2Coordinate().SetValue(*size)
    pl.add_actor(leg, reset_camera=False)
    return leg


def build_plotter(cs, t0, off_screen, geom_label):
    tmax = cs.tmax
    times = cs.paper["times"]
    pl = pv.Plotter(shape=(1, 2), off_screen=off_screen,
                    window_size=(1900, 950), border=False)
    pl.set_background(SURFACE)
    tool = tool_surface()
    inlet, vent = cs.port_meshes()
    title = (f"Double dome, {cs.case['label']} sample, "
             f"{'shear-dependent' if cs.variant == 'shear' else 'isotropic'} "
             f"K, {geom_label}")

    # ---- right: shear map + K1 directions (static) ----
    pl.subplot(0, 1)
    pl.add_mesh(tool, color=TOOL, smooth_shading=True, **SHADING)
    pl.add_mesh(cs.poly, scalars="shear angle [deg]", cmap="Blues",
                clim=(0, 40), **SHADING,
                scalar_bar_args=dict(title="|shear angle| [deg]", color=INK,
                                     vertical=True, position_x=0.88,
                                     position_y=0.25, height=0.5, width=0.05,
                                     fmt="%.0f"))
    pl.add_mesh(cs.k1_lines, color=INK, line_width=1.5)
    pl.add_text("Draped ply: shear angle and K1 direction (Eq. 11)",
                position="upper_left", font_size=11, color=INK)
    add_legend_box(pl, [("K1 principal permeability direction", INK,
                         LINE_FACE),
                        ("tool surface", TOOL, AREA_FACE)],
                   position=(0.55, 0.84), size=(0.43, 0.08))

    # ---- left: filling state (updated by the slider) ----
    pl.subplot(0, 0)
    pl.add_mesh(tool, color=TOOL, smooth_shading=True, **SHADING)
    top = float(vdd.tool_height(0.0, 0.0)[0])
    pl.add_mesh(pv.Sphere(radius=10.0, center=(0.0, 0.0, top + 12.0)),
                color=EXP, smooth_shading=True)
    pl.add_mesh(inlet.extract_feature_edges(boundary_edges=True,
                                            feature_edges=False,
                                            manifold_edges=False),
                color=EXP, line_width=4)
    pl.add_mesh(vent, color=VENT, **SHADING)
    add_legend_box(pl, [
        ("simulated flow front (RTMsim-Py)", INK, LINE_FACE),
        ("measured flow front, Pierce & Falzon 2017", EXP, LINE_FACE),
        ("inlet (50 mm diameter)", EXP, DOT_FACE),
        ("vent, held at 0.3 kPa (never fills)", VENT, AREA_FACE),
        ("dry ply", DRY, AREA_FACE),
        ("tool surface", TOOL, AREA_FACE)],
        position=(0.55, 0.72), size=(0.43, 0.2))
    if not off_screen:
        pl.add_text("keys 1-4: jump to the measured times",
                    position="lower_right", font_size=9, color=MUTED)

    state = {"t": t0}

    def update(t):
        t = float(t)
        state["t"] = t
        pl.subplot(0, 0)
        filled = cs.filled_at(t)
        if filled.any():
            pl.add_mesh(cs.poly.extract_cells(np.where(filled)[0]),
                        scalars="fill time [s]", cmap=FILL_TIME_CMAP,
                        clim=(0.0, tmax), **SHADING, name="filled",
                        scalar_bar_args=dict(title="fill time [s]",
                                             color=INK, vertical=True,
                                             position_x=0.88, position_y=0.25,
                                             height=0.5, width=0.05,
                                             fmt="%.0f"))
        else:
            pl.remove_actor("filled")
        dry = ~filled & ~cs.vent
        if dry.any():
            pl.add_mesh(cs.poly.extract_cells(np.where(dry)[0]),
                        color=DRY, **SHADING, name="dry")
        else:
            pl.remove_actor("dry")
        front = cs.front_lines(filled)
        if front is not None:
            pl.add_mesh(front, color=INK, line_width=4,
                        render_lines_as_tubes=True, name="front")
        else:
            pl.remove_actor("front")
        t_exp = min(times, key=lambda tm: abs(tm - t))
        if t <= 1.1 * times[-1]:
            pl.add_mesh(drape_plan_curve(cs.paper["exp"][t_exp]), color=EXP,
                        line_width=4, render_lines_as_tubes=True, name="exp")
            exp_txt = f"measured front shown: t = {t_exp} s"
        else:
            pl.remove_actor("exp")
            exp_txt = "past the last measured time, no measured front"
        pct = 100.0 * filled[~cs.vent].mean()
        pl.add_text(f"{title}\nt = {t:.0f} s, {pct:.0f} % of the ply "
                    f"(without vents) filled; {exp_txt}",
                    position="upper_left", font_size=11, color=INK,
                    name="time_text")

    if not off_screen:
        slider = pl.add_slider_widget(
            update, [0.0, tmax], value=t0, title="time [s]", fmt="%.0f",
            pointa=(0.08, 0.1), pointb=(0.45, 0.1), color=INK,
            style="modern")

        def jump(k):
            def _cb():
                slider.GetRepresentation().SetValue(times[k])
                update(times[k])
            return _cb
        for k in range(len(times)):
            pl.add_key_event(str(k + 1), jump(k))
    update(t0)

    for k in (0, 1):
        pl.subplot(0, k)
        pl.view_isometric()
        pl.camera.azimuth = -30
        pl.camera.elevation = 15
        pl.reset_camera()
        pl.camera.zoom(1.25)
    pl.link_views()
    pl.add_axes(color=INK)
    return pl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--case", choices=list(vdd.CASES), default="0_90")
    ap.add_argument("--variant", choices=("shear", "iso"), default="shear")
    ap.add_argument("--geometry", choices=("stl", "parametric"), default="stl")
    ap.add_argument("--stl", default=vdd.STL_DEFAULT)
    ap.add_argument("--spacing", type=float, default=10.0)
    ap.add_argument("--time", type=float, default=None,
                    help="initial / screenshot time [s] "
                         "(default: last measured time)")
    ap.add_argument("--tmax", type=float, default=None,
                    help="simulated time [s] (default: just past the last "
                         "measured time, 1900 s for 0/90, 1450 s for 45)")
    ap.add_argument("--off-screen", action="store_true",
                    help="render to PNG only (no window)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.geometry == "stl":
        vdd._TOOL = vdd.StlTool(args.stl)
    geom_label = ("benchmark STL tool x2" if args.geometry == "stl"
                  else "parametric tool")
    cs = Case(args.case, args.variant, args.spacing, args.tmax)
    t0 = args.time if args.time is not None else cs.paper["times"][-1]

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output_validation_double_dome", args.geometry)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"view_{args.case}_{args.variant}.png")
    pl = build_plotter(cs, t0, args.off_screen, geom_label)
    if args.off_screen:
        pl.screenshot(out)
        pl.close()
    else:
        pl.show(screenshot=out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
