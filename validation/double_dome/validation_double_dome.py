"""
RTMsim-Py validation: vacuum infusion of a draped plain-weave ply over a
double dome tool, against the experiments of

    R.S. Pierce, B.G. Falzon, "Simulating Resin Infusion through Textile
    Reinforcement Materials for the Manufacture of Complex Composite
    Structures", Engineering 3 (2017) 596-607 (validation/ folder),
    sec. 5 and Figs. 7-13; experimental details also in Pierce, Falzon,
    Thompson, ICCM20 (2015), "Complete process model for manufacturing
    complex composite structures with fabric reinforcements".

Test case (from the papers)
  * single ply of dry carbon plain weave, 800 x 500 mm, 0.4 mm thick,
    porosity 0.724 undeformed, draped over a male double dome tool
    120 mm high (top flush with a 950 x 550 mm frame)
  * olive oil through a 50 mm diameter central inlet on top of the dome,
    p_inlet = 101.3 kPa, vacuum outlet at the long ends p = 0.3 kPa
  * viscosity 0.0756 Pa s (0/90 sample) and 0.0993 Pa s (-45/45 sample)
  * quarter symmetry, free-slip symmetry walls, outlet on the far end
  * "Basic" model: isotropic K = 3.3e-11 m^2
  * shear-dependent model: Eqs. (9)-(11) of the paper, K1/K2 from the
    local shear angle, K1 direction from the weft towards the bias
  * measured flow fronts: 50/580/1255/1795 s (0/90) and
    20/300/850/1340 s (-45/45), Figs. 12 and 13 (digitised below)

What this script adds (not given in the paper)
  * Tool surface (--geometry stl, default). The double dome benchmark die
    (validation/double_dome/*.stl, 270 x 470 x 60 mm) inverted into the
    male tool and scaled x2 to 540 x 940 x 120 mm, which matches the
    paper's 950 x 550 mm frame with the tool top 120 mm deep.
  * Tool surface (--geometry parametric), used before the STL was
    available. A parametric ridge with rounded ends: flat top of half-width
    A_TOP, a cosine side wall that drops H = 120 mm over a plan run
    R_SLOPE, floor beyond. A_TOP and R_SLOPE were fitted (residuals
    <= 2 mm) to six independent data of the paper: the isotropic
    "Basic" fronts along the short axis of both samples (Figs. 12, 13:
    surface radius from flat radial Darcy flow vs. plan position) and
    the ply draw-in (250 mm of fabric seen at 190 mm in plan). The
    flat top is long enough along the long axis that the "Basic" fronts
    there follow flat radial flow up to 1795 s, as in the paper.
  * Draping. The paper used an Abaqus continuum draping model; here a
    kinematic (fishnet) drape on that surface gives the per-element
    shear angle and yarn directions; generators are the surface
    geodesics from the inlet along the yarn directions.

Outputs (output_validation_double_dome/<geometry>/)
  verification_flat_plate.png  solver vs analytical radial flow (flat)
  drape_shear.png              draped ply, shear angle field (both samples)
  fronts_0_90.png, fronts_45.png   flow fronts at the measured times
  front_vs_time.png            front position on both symmetry axes vs time
  summary.csv                  front positions, errors and front speeds

Run:  python validation_double_dome.py [--geometry stl|parametric]
      [--skip-verification] [--spacing 10] [--model 1|2|4]
      --model 4 (incompressible resin) adds a third variant, shear-dependent
      K with the K1 angle measured from the warp instead of the weft, and a
      run of every variant to the end of fill; outputs go to <geometry>_model4
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))   # repo root (rtmsim.py)
import rtmsim as rtm


# --------------------------------------------------------------------------
# Paper inputs (Pierce & Falzon 2017, sec. 5.1-5.3)
# --------------------------------------------------------------------------
P_INLET = 101.3e3          # [Pa]
P_OUTLET = 0.3e3           # [Pa], also the initial cavity pressure
PHI_0 = 0.724              # undeformed porosity
THICKNESS = 0.4e-3         # [m]
K_BASIC = 3.3e-11          # [m^2], isotropic "Basic" model
INLET_RADIUS = 25.0        # [mm], 50 mm diameter inlet
PLY_HALF = (250.0, 400.0)  # [mm], quarter of the 500 x 800 mm ply (x, y)
RHO_OIL = 910.0            # [kg/m^3], olive oil (not given; enters the
                           # inertia term only, negligible in Darcy flow)
CASES = {
    "0_90": dict(label="0°/90° (warp along y)", fibre_deg=0.0, mu=0.0756,
                 tmax=1900.0),
    "45":   dict(label="−45°/45°", fibre_deg=45.0, mu=0.0993, tmax=1450.0),
}

# --------------------------------------------------------------------------
# Tool surface (fitted, see module docstring)  [mm]
# --------------------------------------------------------------------------
H_TOOL = 120.0     # tool height
A_TOP = 27.0       # half-width of the flat top along the short axis
R_SLOPE = 102.6    # plan run of the cosine side wall (61 deg max slope)
Y_CAP = 305.0      # half-length of the straight part of the ridge; flat
                   # top on the long axis ends at Y_CAP + A_TOP = 332 mm


STL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "DoubleDome - Tooling -remeshed.stl")
STL_SCALE = 2.0    # benchmark die 270 x 470 x 60 mm -> 540 x 940 x 120 mm,
                   # i.e. the paper's 550 x 950 mm frame, 120 mm deep


class StlTool:
    """
    Male tool height field from the double dome benchmark die STL
    (Y up, Z along the long axis, cavity from Y = -60 to 0). The cavity is
    inverted into the male tool, scaled by STL_SCALE, sampled by ray
    casting, made quarter-symmetric (as the paper's model) and fitted
    with a bicubic spline for a smooth height and gradient.
    """

    def __init__(self, path, scale=STL_SCALE, step=1.0):
        import trimesh
        from scipy.interpolate import RectBivariateSpline
        from scipy.ndimage import gaussian_filter
        m = trimesh.load(path, force="mesh")
        (x0, y0, z0), (x1, y1, z1) = m.bounds
        xs = np.arange(0.0, min(-x0, x1) + 1e-9, step)
        zs = np.arange(0.0, min(-z0, z1) + 1e-9, step)
        xs = np.concatenate([-xs[:0:-1], xs])
        zs = np.concatenate([-zs[:0:-1], zs])
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        o = np.column_stack([X.ravel(), np.full(X.size, y1 + 10.0), Z.ravel()])
        d = np.tile([0.0, -1.0, 0.0], (X.size, 1))
        loc, ray, _ = m.ray.intersects_location(o, d, multiple_hits=False)
        hit = np.full(X.size, np.nan)
        hit[ray] = loc[:, 1]
        hit = hit.reshape(X.shape)
        if np.isnan(hit).any():         # rays through tiny gaps: nearest
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(np.isnan(hit), return_distances=False,
                                         return_indices=True)
            hit = hit[tuple(idx)]
        h = (y1 - hit) * scale          # cavity bottom -> male tool top
        h = 0.25 * (h + h[::-1] + h[:, ::-1] + h[::-1, ::-1])
        h = gaussian_filter(h, 1.0)     # remove facet kinks before the spline
        self.height = float(h.max())
        self.spline = RectBivariateSpline(xs * scale, zs * scale, h)
        self.extent = (xs[-1] * scale, zs[-1] * scale)

    def __call__(self, x, y):
        x = np.clip(np.asarray(x, dtype=np.float64), -self.extent[0],
                    self.extent[0])
        y = np.clip(np.asarray(y, dtype=np.float64), -self.extent[1],
                    self.extent[1])
        s = self.spline
        return (s(x, y, grid=False), s(x, y, dx=1, grid=False),
                s(x, y, dy=1, grid=False))


_TOOL = None       # StlTool when --geometry stl, else the parametric ridge


def tool_height(x, y):
    """z(x, y) of the tool and its gradient (dz/dx, dz/dy), plan in mm."""
    if _TOOL is not None:
        return _TOOL(x, y)
    return parametric_height(x, y)


def parametric_height(x, y):
    """Fitted parametric ridge (see module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ey = np.maximum(np.abs(y) - Y_CAP, 0.0)
    dp = np.hypot(x, ey)
    dd = dp - A_TOP
    t = np.clip(dd / R_SLOPE, 0.0, 1.0)
    z = 0.5 * H_TOOL * (1.0 + np.cos(np.pi * t))
    on_wall = (dd > 0.0) & (dd < R_SLOPE)
    dz_dd = np.where(on_wall,
                     -0.5 * H_TOOL * np.pi / R_SLOPE * np.sin(np.pi * t), 0.0)
    safe = np.where(dp > 0.0, dp, 1.0)
    return z, dz_dd * x / safe, dz_dd * np.sign(y) * ey / safe


def surface_point(xy):
    z, _, _ = tool_height(xy[..., 0], xy[..., 1])
    return np.concatenate([xy, z[..., None]], axis=-1)


# --------------------------------------------------------------------------
# Kinematic (fishnet) drape
# --------------------------------------------------------------------------
def geodesic(direction_xy, n, d):
    """n+1 points spaced d along the surface geodesic leaving the inlet."""
    u0 = np.asarray(direction_xy, dtype=np.float64)
    u0 = u0 / np.linalg.norm(u0)
    pts = [surface_point(np.zeros(2))]
    for k in range(n):
        Pk = pts[-1]
        if k == 0:
            u = u0
        else:
            # Discrete geodesic: the kink 2 P_k - P_(k-1) -> surface is
            # along the surface normal at P_k.
            Q = 2.0 * Pk - pts[-2]
            _, hx, hy = tool_height(Pk[0], Pk[1])
            nrm = np.array([-hx, -hy, 1.0]) / np.sqrt(hx * hx + hy * hy + 1)
            lam = 0.0
            for _ in range(30):
                q = Q + lam * nrm
                zq, qx, qy = tool_height(q[0], q[1])
                g = q[2] - zq
                dg = nrm[2] - (qx * nrm[0] + qy * nrm[1])
                lam -= g / dg
                if abs(g) < 1e-10:
                    break
            q = Q + lam * nrm
            u = q[:2] - Pk[:2]
            u = u / np.linalg.norm(u)
        # Step length s in plan so that the 3-D chord is exactly d.
        s = d
        for _ in range(30):
            xy = Pk[:2] + s * u
            zz, hx, hy = tool_height(xy[0], xy[1])
            D = np.array([s * u[0], s * u[1], zz - Pk[2]])
            f = D @ D - d * d
            df = 2.0 * (D[0] * u[0] + D[1] * u[1]
                        + D[2] * (hx * u[0] + hy * u[1]))
            s -= f / df
            if abs(f) < 1e-10 * d * d:
                break
        pts.append(surface_point(Pk[:2] + s * u))
    return np.array(pts)


def fishnet_quadrant(g1, g2, d):
    """Fill the net spanned by generators g1 (index i) and g2 (index j)."""
    n1, n2 = len(g1) - 1, len(g2) - 1
    P = np.full((n1 + 1, n2 + 1, 3), np.nan)
    P[:, 0] = g1
    P[0, :] = g2
    for k in range(2, n1 + n2 + 1):
        i = np.arange(max(1, k - n2), min(n1, k - 1) + 1)
        j = k - i
        A, B, C = P[i - 1, j], P[i, j - 1], P[i - 1, j - 1]
        ok = np.all(np.isfinite(A) & np.isfinite(B) & np.isfinite(C), axis=1)
        xy = A[:, :2] + B[:, :2] - C[:, :2]
        xy[~ok] = 0.0
        for _ in range(40):
            z, hx, hy = tool_height(xy[:, 0], xy[:, 1])
            ra = np.column_stack([xy - A[:, :2], z - A[:, 2]])
            rb = np.column_stack([xy - B[:, :2], z - B[:, 2]])
            F1 = np.einsum("ij,ij->i", ra, ra) - d * d
            F2 = np.einsum("ij,ij->i", rb, rb) - d * d
            J11 = 2 * (ra[:, 0] + ra[:, 2] * hx)
            J12 = 2 * (ra[:, 1] + ra[:, 2] * hy)
            J21 = 2 * (rb[:, 0] + rb[:, 2] * hx)
            J22 = 2 * (rb[:, 1] + rb[:, 2] * hy)
            det = J11 * J22 - J12 * J21
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dx = (J22 * F1 - J12 * F2) / det
            dy = (-J21 * F1 + J11 * F2) / det
            step = np.hypot(dx, dy)
            scale = np.minimum(1.0, 0.5 * d / np.maximum(step, 1e-30))
            xy[:, 0] -= scale * dx
            xy[:, 1] -= scale * dy
        z, _, _ = tool_height(xy[:, 0], xy[:, 1])
        Pn = np.column_stack([xy, z])
        res = np.maximum(np.abs(np.linalg.norm(Pn - A, axis=1) - d),
                         np.abs(np.linalg.norm(Pn - B, axis=1) - d))
        ok &= res < 1e-6 * d
        Pn[~ok] = np.nan
        P[i, j] = Pn
    return P


def drape_ply(fibre_deg, d):
    """
    Drape the quarter ply (material x in [0, 250], y in [0, 400] mm) with
    yarns at +fibre_deg (weft, index i) and fibre_deg + 90 (warp, index j).
    Returns plan/3-D nodes, triangles, material coords and per-triangle
    shear angle [deg], weft and warp unit vectors.
    """
    a = np.radians(fibre_deg)
    f1 = np.array([np.cos(a), np.sin(a)])          # weft
    f2 = np.array([-np.sin(a), np.cos(a)])         # warp
    Xmax, Ymax = PLY_HALF
    # Material coords of net node (i, j): X = d (i f1 + j f2).
    corners = np.array([[0, 0], [Xmax, 0], [0, Ymax], [Xmax, Ymax]])
    ij = corners @ np.column_stack([f1, f2]) / d
    i_hi = int(np.ceil(ij[:, 0].max()))
    j_lo, j_hi = int(np.floor(ij[:, 1].min())), int(np.ceil(ij[:, 1].max()))
    g1 = geodesic(f1, i_hi, d)
    nodes = np.full((i_hi + 1, j_hi - j_lo + 1, 3), np.nan)
    if j_hi > 0:
        nodes[:, -j_lo:] = fishnet_quadrant(g1, geodesic(f2, j_hi, d), d)
    if j_lo < 0:
        Pm = fishnet_quadrant(g1, geodesic(-f2, -j_lo, d), d)
        nodes[:, :-j_lo + 1] = Pm[:, ::-1]
    I, Jn = np.meshgrid(np.arange(i_hi + 1), np.arange(j_lo, j_hi + 1),
                        indexing="ij")
    mat = d * (I[..., None] * f1 + Jn[..., None] * f2)

    nid = np.arange(nodes.shape[0] * nodes.shape[1]).reshape(nodes.shape[:2])
    tris, quad_of = [], []
    for ii in range(nodes.shape[0] - 1):
        for jj in range(nodes.shape[1] - 1):
            q = nodes[ii:ii + 2, jj:jj + 2]
            if not np.all(np.isfinite(q)):
                continue
            n00, n10 = nid[ii, jj], nid[ii + 1, jj]
            n01, n11 = nid[ii, jj + 1], nid[ii + 1, jj + 1]
            # Split along the diagonal that follows the material symmetry
            # line of that half of the net, so x = 0 and y = 0 are exact.
            if Jn[ii, jj] >= 0:
                cand = [(n00, n10, n11), (n00, n11, n01)]
            else:
                cand = [(n00, n10, n01), (n10, n11, n01)]
            for t in cand:
                tris.append(t)
                quad_of.append((ii, jj))
    tris = np.array(tris)
    quad_of = np.array(quad_of)
    flat_mat = mat.reshape(-1, 2)
    cm = flat_mat[tris].mean(axis=1)
    eps = 1e-6
    keep = ((cm[:, 0] > -eps) & (cm[:, 1] > -eps)
            & (cm[:, 0] < Xmax + eps) & (cm[:, 1] < Ymax + eps))
    tris, quad_of = tris[keep], quad_of[keep]

    # Shear angle and yarn directions per quad (mean of opposite edges).
    qi, qj = quad_of[:, 0], quad_of[:, 1]
    weft = (nodes[qi + 1, qj] - nodes[qi, qj]
            + nodes[qi + 1, qj + 1] - nodes[qi, qj + 1])
    warp = (nodes[qi, qj + 1] - nodes[qi, qj]
            + nodes[qi + 1, qj + 1] - nodes[qi + 1, qj])
    weft /= np.linalg.norm(weft, axis=1, keepdims=True)
    warp /= np.linalg.norm(warp, axis=1, keepdims=True)
    shear = 90.0 - np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", weft, warp), -1, 1)))

    used, inv = np.unique(tris.ravel(), return_inverse=True)
    xyz = nodes.reshape(-1, 3)[used]
    return dict(xyz=xyz, tris=inv.reshape(-1, 3), mat=flat_mat[used],
                shear=shear, weft=weft, warp=warp, spacing=d)


# --------------------------------------------------------------------------
# Deformation-dependent permeability, Eqs. (9)-(11) of the paper
# --------------------------------------------------------------------------
SHEAR_MAX = 40.0   # [deg], upper end of the permeability characterisation


def K1_of_shear(g_deg):
    g = np.radians(np.abs(g_deg))
    return 0.667 * (-6.641 * g**4 + 13.28 * g**3 - 8.414 * g**2
                    + 2.4 * g + 0.6028) * 1e-10


def K2_of_shear(g_deg):
    g = np.radians(np.abs(g_deg))
    return 0.5 * (-7.7 * g**4 + 14.66 * g**3 - 9.261 * g**2
                  + 1.605 * g + 0.5313) * 1e-10


def principal_angle(g_deg):
    """Eq. (11): angle [deg] of K1 from the weft towards the bias."""
    g = np.abs(g_deg)
    return np.where(g <= 20.0, g / 20.0 * (45.0 - g / 2.0), 45.0 - g / 2.0)


def porosity_of_shear(g_deg):
    """Areal density rises as 1/cos(gamma) under shear at constant thickness."""
    return 1.0 - (1.0 - PHI_0) / np.cos(np.radians(g_deg))


def k1_direction(drape, k1_on="weft"):
    """
    Unit K1 vectors: weft rotated by Eq. (11) towards the acute bisector.
    The paper does not say which yarn of the 0/90 ply is the weft;
    k1_on="warp" starts from the other yarn instead.
    """
    w1, w2 = drape["weft"], drape["warp"]
    if k1_on == "warp":
        w1, w2 = w2, w1
    flip = np.where(np.einsum("ij,ij->i", w1, w2) >= 0.0, 1.0, -1.0)
    b = w1 + flip[:, None] * w2
    t = b - np.einsum("ij,ij->i", b, w1)[:, None] * w1
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(tn > 1e-12, t / np.maximum(tn, 1e-30), w2)
    phi = np.radians(principal_angle(drape["shear"]))[:, None]
    return np.cos(phi) * w1 + np.sin(phi) * t


# --------------------------------------------------------------------------
# Simulation set-up
# --------------------------------------------------------------------------
AIR_EOS = dict(p_ref=1.01325e5, rho_ref=1.225, gamma=1.4)


VARIANTS = {   # name: (shear-dependent K, yarn the K1 angle starts from)
    "iso": (False, None),
    "shear": (True, "weft"),
    "shear_warp": (True, "warp"),
}


def build_sim(mesh, drape, mu, tmax, model, shear_dependent, k1_on="weft"):
    resin = (rtm.ResinMaterial("olive oil").set_viscosity(mu)
             .set_density(RHO_OIL))
    t_ply = THICKNESS
    sim = (rtm.RTMSimulation().set_mesh(mesh).set_process_model(model)
           .set_resin(resin)
           .set_pressures(p_inlet=P_INLET, p_init=P_OUTLET)
           .set_air_eos(**AIR_EOS)
           .set_run_control(tmax=tmax, n_pics=20)
           .add_injection_port((0.0, 0.0, H_TOOL), radius=INLET_RADIUS))
    if not shear_dependent:
        fab = (rtm.FabricMaterial("plain weave, isotropic")
               .set_permeability(K_BASIC, K_BASIC).set_porosity(PHI_0))
        sim.set_laminate(rtm.LaminateStack().add_ply(fab, t_ply,
                                                     refdir=(1, 0, 0)))
    else:
        # One stack per 1 deg shear class, K1 along the per-element
        # principal direction of Eq. (11). Eqs. (9)-(10) were characterised
        # for 0-40 deg; the fishnet locks harder than that at the far end of
        # the ridge (no draw-in), so the shear is clamped to 40 deg there.
        e1 = k1_direction(drape, k1_on)
        g_bin = np.round(np.minimum(np.abs(drape["shear"]), SHEAR_MAX)
                         ).astype(int)
        for k, gb in enumerate(np.unique(g_bin)):
            fab = (rtm.FabricMaterial(f"plain weave, shear {gb} deg")
                   .set_permeability(float(K1_of_shear(gb)),
                                     float(K2_of_shear(gb)))
                   .set_porosity(float(porosity_of_shear(gb))))
            stack = rtm.LaminateStack().add_ply(fab, t_ply, refdir=e1)
            cells = np.where(g_bin == gb)[0]
            if k == 0:
                sim.set_laminate(stack)
            sim.add_laminate_region(stack, cells)
    # Vacuum outlet: last row of elements at the far (long) end of the ply.
    ymat = drape["mat"][drape["tris"]].mean(axis=1)[:, 1]
    sim.add_vent_cells(np.where(ymat > PLY_HALF[1] - drape["spacing"])[0])
    return sim


def full_fill(mesh, drape, case):
    """i_model=4: run each variant on to the end of fill (ply edges, vents)."""
    for variant, (shear_dep, k1_on) in VARIANTS.items():
        sim = build_sim(mesh, drape, case["mu"], 20.0 * case["tmax"], 4,
                        shear_dep, k1_on)
        sim.run()
        g = sim.get_final_snapshot().gamma
        vents = np.concatenate([p.cells for p in sim.get_ports()
                                if p.kind == "vent"])
        spots = sim.get_dry_spots()
        print(f"    full fill, {variant}: "
              f"{'complete' if sim.is_fill_complete() else 'NOT complete'} "
              f"at {sim.get_total_fill_time():.0f} s; cells not full "
              f"{int((g < 1).sum())} of {g.size} (vents {int((g[vents] < 1).sum())}"
              f" of {vents.size}); {len(spots)} dry spot(s)"
              + (f", air {sum(d['air_volume'] for d in spots):.2e} m^3"
                 if spots else ""))


def run_sim(sim, label):
    t0 = time.time()
    sim.run()
    status = "complete" if sim.is_fill_complete() else "tmax reached"
    print(f"    {label}: {time.time() - t0:.1f} s wall, {status}")
    return sim.get_fill_time_field()


# --------------------------------------------------------------------------
# Digitised data, Pierce & Falzon 2017 Figs. 12/13 (plan view, mm, +-3 mm;
# 0/90 read on the gridded copy, Fig. 7 of the ICCM20 paper)
# --------------------------------------------------------------------------
PAPER = {
    "0_90": {
        "times": [50, 580, 1255, 1795],
        "exp": {
            50: [(0, 112), (20, 110), (35, 105), (48, 97), (58, 85),
                 (66, 65), (70, 40), (72, 15), (72, 0)],
            580: [(0, 228), (35, 229), (60, 226), (100, 218), (120, 205),
                  (136, 190), (150, 162), (158, 125), (162, 76), (166, 20),
                  (166, 0)],
            1255: [(0, 288), (50, 286), (60, 292), (75, 305), (90, 318),
                   (110, 325), (140, 332), (155, 330), (180, 320),
                   (210, 305), (228, 298)],
            1795: [(0, 322), (15, 333), (30, 350), (50, 368), (75, 383),
                   (100, 393), (130, 400), (150, 402), (180, 400),
                   (200, 396), (220, 390), (245, 384)],
        },
        "model": {
            50: [(0, 118), (15, 118), (30, 114), (48, 104), (56, 92),
                 (63, 73), (68, 45), (70, 0)],
            580: [(0, 225), (45, 222), (75, 210), (90, 203), (105, 190),
                  (115, 175), (130, 155), (145, 135), (160, 115), (170, 95),
                  (176, 65), (178, 30), (178, 0)],
            1255: [(0, 290), (50, 289), (80, 290), (100, 296), (125, 303),
                   (140, 302), (160, 295), (180, 285), (195, 277),
                   (205, 262), (208, 240)],
            1795: [(0, 330), (20, 335), (50, 345), (80, 355), (110, 365),
                   (150, 376), (180, 373), (210, 368), (230, 362),
                   (242, 355)],
        },
        "basic": {
            50: [(0, 85), (15, 84), (32, 79), (45, 72), (55, 61), (62, 40),
                 (65, 20), (66, 0)],
            580: [(0, 208), (30, 205), (55, 198), (65, 195), (80, 183),
                  (90, 170), (96, 155), (110, 135), (125, 110), (135, 85),
                  (142, 70), (148, 40), (150, 0)],
            1255: [(0, 280), (30, 278), (50, 272), (80, 260), (100, 250),
                   (130, 235), (150, 222), (170, 205), (180, 192),
                   (187, 172), (190, 145), (191, 120)],
            1795: [(0, 320), (30, 320), (60, 318), (90, 314), (120, 308),
                   (150, 300), (180, 290), (200, 280), (212, 265)],
        },
        "outline": [(0, 400), (50, 405), (100, 420), (135, 433), (150, 436),
                    (200, 440), (245, 440), (245, 350), (240, 330),
                    (225, 280), (215, 255), (205, 215), (200, 185),
                    (197, 160), (193, 145), (191, 100), (191, 0)],
    },
    "45": {
        "times": [20, 300, 850, 1340],
        "exp": {
            20: [(0, 87), (10, 91), (25, 93), (40, 90), (50, 76), (55, 55),
                 (57, 30), (57, 0)],
            300: [(0, 203), (15, 206), (30, 207), (55, 203), (75, 195),
                  (85, 180), (95, 165), (102, 125), (105, 60), (105, 0)],
            850: [(0, 270), (30, 273), (50, 272), (80, 262), (100, 250),
                  (120, 232), (130, 215), (150, 205), (170, 192), (180, 175),
                  (187, 140), (190, 120), (188, 80), (188, 0)],
            1340: [(0, 302), (30, 306), (60, 303), (90, 297), (120, 286),
                   (150, 268), (170, 258), (190, 252), (212, 243)],
        },
        "model": {
            20: [(0, 108), (15, 109), (25, 108), (40, 95), (48, 80),
                 (52, 55), (53, 0)],
            300: [(0, 205), (25, 207), (50, 202), (70, 193), (80, 180),
                  (88, 160), (95, 130), (98, 100), (100, 50), (100, 0)],
            850: [(0, 270), (30, 273), (50, 272), (80, 262), (100, 247),
                  (120, 228), (135, 212), (150, 190), (160, 165),
                  (176, 140), (178, 100), (172, 60), (168, 30), (165, 0)],
            1340: [(0, 310), (30, 315), (60, 309), (90, 303), (120, 298),
                   (150, 292), (170, 280), (185, 262), (200, 240),
                   (205, 204), (195, 150), (187, 94), (185, 50), (184, 0)],
        },
        "basic": {
            20: [(0, 61), (15, 61), (30, 57), (40, 50), (48, 40), (53, 20),
                 (54, 0)],
            300: [(0, 148), (25, 148), (45, 140), (60, 128), (75, 105),
                  (85, 80), (92, 55), (96, 30), (98, 0)],
            850: [(0, 220), (50, 212), (85, 193), (118, 149), (140, 105),
                  (151, 72), (157, 39), (157, 0)],
            1340: [(0, 259), (69, 243), (124, 199), (162, 149), (179, 116),
                   (187, 39), (187, 0)],
        },
        "outline": [(0, 400), (50, 397), (100, 390), (130, 383), (150, 377),
                    (180, 370), (210, 371), (212, 300), (210, 250),
                    (200, 200), (193, 150), (188, 100), (185, 50), (184, 0)],
    },
}


def radial_fill_time(r_mm, mu):
    """Flat radial Darcy flow from the inlet radius, constant pressure."""
    r = np.asarray(r_mm) * 1e-3
    r0 = INLET_RADIUS * 1e-3
    c = PHI_0 * mu / (K_BASIC * (P_INLET - P_OUTLET))
    return c * (r**2 / 2 * np.log(r / r0) - (r**2 - r0**2) / 4)


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
def nodal_fill_time(tris, n_nodes, ft, t_dry):
    ftc = np.where(np.isfinite(ft), ft, t_dry)
    acc = np.zeros(n_nodes)
    cnt = np.zeros(n_nodes)
    for k in range(3):
        np.add.at(acc, tris[:, k], ftc)
        np.add.at(cnt, tris[:, k], 1.0)
    return acc / np.maximum(cnt, 1)


def front_contour(tri, fnode, t):
    """Plan-view iso-line fill_time = t as a list of (n, 2) arrays."""
    fig = plt.figure()
    cs = plt.tricontour(tri, fnode, levels=[t])
    segs = [s for s in cs.allsegs[0] if len(s) > 1]
    plt.close(fig)
    return segs


def polar_profile(segs, theta):
    """Front radius r(theta) from the inlet, NaN outside the curve's span."""
    if not segs:
        return np.full_like(theta, np.nan)
    pts = np.vstack(segs)
    th = np.arctan2(pts[:, 1], pts[:, 0])
    r = np.hypot(pts[:, 0], pts[:, 1])
    o = np.argsort(th)
    th, r = th[o], r[o]
    out = np.interp(theta, th, r)
    out[(theta < th.min() - 1e-9) | (theta > th.max() + 1e-9)] = np.nan
    return out


def axis_front(plan, line_nodes, fnode, coord, times):
    """Front position on a symmetry axis at the given times (plan mm)."""
    pos = plan[line_nodes, coord]
    o = np.argsort(pos)
    pos, ft = pos[o], np.maximum.accumulate(fnode[line_nodes][o])
    out = np.interp(times, ft, pos, right=np.nan)
    return out


def curve_axis_values(curve):
    """(front on y axis, front on x axis or NaN if it ends on the edge)."""
    c = np.asarray(curve, dtype=float)
    y_axis = c[0, 1] if c[0, 0] == 0 else np.nan
    x_axis = c[-1, 0] if c[-1, 1] == 0 else np.nan
    return y_axis, x_axis


# --------------------------------------------------------------------------
# Plot styling (reference palette: slot 1 blue, slot 2 orange; experiment
# in primary ink)
# --------------------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # shear-dependent model (ours solid, paper dashed)
ORANGE = "#eb6834"   # isotropic model (ours solid, paper dashed)
AQUA = "#1baf7a"     # shear-dependent model, K1 angle from the warp (ours)
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9, "legend.frameon": False,
})


def style_plan_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 450)
    ax.set_xlabel("distance from inlet, x [mm]")


# --------------------------------------------------------------------------
# Verification: flat plate, analytical radial flow
# --------------------------------------------------------------------------
def verification(outdir, spacing, model):
    print("\n[1] Verification: flat plate radial flow vs analytical")
    n_div = int(round(800.0 / spacing))
    mesh = rtm.ShellMesh.make_square_plate(side=0.8, n_div=n_div)
    cc = mesh.cellcenter
    vent = np.where(np.max(np.abs(cc[:, :2]), axis=1) > 0.4 - 0.8 / n_div)[0]
    fab = (rtm.FabricMaterial("isotropic").set_permeability(K_BASIC, K_BASIC)
           .set_porosity(PHI_0))
    mu = CASES["0_90"]["mu"]
    rows = []
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    r_an = np.linspace(INLET_RADIUS + 1, 330, 200)
    ax.plot(r_an, radial_fill_time(r_an, mu), color=MUTED, lw=1.2,
            label="analytical radial Darcy flow")
    for m, color in ((model, BLUE), (1, ORANGE)):
        sim = (rtm.RTMSimulation().set_mesh(mesh).set_process_model(m)
               .set_resin(rtm.ResinMaterial("oil").set_viscosity(mu)
                          .set_density(RHO_OIL))
               .set_laminate(rtm.LaminateStack().add_ply(fab, THICKNESS,
                                                         refdir=(1, 0, 0)))
               .set_pressures(p_inlet=P_INLET, p_init=P_OUTLET)
               .set_air_eos(**AIR_EOS)
               .set_run_control(tmax=1800.0, n_pics=20)
               .add_injection_port((0, 0, 0), radius=INLET_RADIUS * 1e-3)
               .add_vent_cells(vent))
        try:
            ft = run_sim(sim, f"i_model={m}, {mesh.N} cells")
        except FloatingPointError as e:
            print(f"    i_model={m}: diverged ({str(e)[:50]}...)")
            continue
        r = np.linalg.norm(cc[:, :2], axis=1) * 1e3
        bins = np.arange(60, 335, 10.0)
        t_med = np.array([np.nanmedian(ft[np.abs(r - b) < spacing / 2])
                          if np.any(np.abs(r - b) < spacing / 2) else np.nan
                          for b in bins])
        ax.plot(bins, t_med, "o", ms=4, color=color, mfc=color, mec=SURFACE,
                mew=0.8, label=f"RTMsim-Py i_model={m}")
        for b in (88, 150, 210, 280, 320):
            sel = np.abs(r - b) < spacing / 2
            ts = float(np.nanmedian(ft[sel]))
            ta = float(radial_fill_time(b, mu))
            rows.append((m, b, ts, ta))
            print(f"      r = {b:3d} mm: t_sim = {ts:7.1f} s, "
                  f"t_analytical = {ta:7.1f} s ({100*(ts/ta-1):+.0f} %)")
    ax.set_xlabel("front radius [mm]")
    ax.set_ylabel("fill time [s]")
    ax.set_title(f"Flat plate, K = {K_BASIC:.1e} m², φ = {PHI_0}, "
                 f"{spacing:g} mm cells", fontsize=9, color=INK2, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "verification_flat_plate.png"), dpi=150)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--spacing", type=float, default=10.0,
                    help="drape net / mesh spacing [mm] (default 10)")
    ap.add_argument("--model", type=int, default=2, choices=(1, 2, 4),
                    help="process model (default 2: constant porosity; "
                         "model 1 ignores porosity; model 4: incompressible "
                         "resin, also runs K1 on the warp and a full fill)")
    ap.add_argument("--geometry", choices=("stl", "parametric"),
                    default="stl",
                    help="tool surface: benchmark STL x2 (default) or the "
                         "parametric ridge fitted to the paper's data")
    ap.add_argument("--stl", default=STL_DEFAULT, help="double dome die STL")
    ap.add_argument("--skip-verification", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output_validation_double_dome", args.geometry
                          + ("" if args.model == 2 else f"_model{args.model}"))
    os.makedirs(outdir, exist_ok=True)
    variants = ["iso", "shear"] + (["shear_warp"] if args.model == 4 else [])

    if not args.skip_verification:
        verification(outdir, args.spacing, args.model)

    global _TOOL
    if args.geometry == "stl":
        _TOOL = StlTool(args.stl)
        print(f"\nTool: {os.path.basename(args.stl)} x{STL_SCALE:g}, "
              f"height {_TOOL.height:.1f} mm")
    print(f"\n[2] Double dome ({args.geometry} tool), i_model={args.model}, "
          f"{args.spacing:g} mm drape net")
    results = {}
    for key, case in CASES.items():
        print(f"  case {case['label']}")
        dr = drape_ply(case["fibre_deg"], args.spacing)
        mesh = rtm.ShellMesh.from_arrays(dr["xyz"], dr["tris"], scale=1e-3,
                                         units="mm")
        g = dr["shear"]
        print(f"    mesh {mesh.N} triangles; shear {g.min():.1f} .. "
              f"{g.max():.1f} deg")
        fts = {}
        for variant in variants:
            shear_dep, k1_on = VARIANTS[variant]
            sim = build_sim(mesh, dr, case["mu"], case["tmax"], args.model,
                            shear_dep, k1_on)
            fts[variant] = run_sim(sim, variant)
        results[key] = dict(drape=dr, ft=fts, case=case)
        if args.model == 4:
            full_fill(mesh, dr, case)

    rows = postprocess(results, outdir)
    write_csv(rows, os.path.join(outdir, "summary.csv"))
    print(f"\nSaved figures and summary.csv to {outdir}")


def symmetry_lines(dr):
    """Node ids on the material lines x = 0 (y axis) and y = 0 (x axis)."""
    m = dr["mat"]
    tol = 1e-6
    return np.where(np.abs(m[:, 0]) < tol)[0], np.where(np.abs(m[:, 1]) < tol)[0]


def postprocess(results, outdir):
    theta = np.radians(np.linspace(3, 87, 85))
    rows = []
    fig_d, axes_d = plt.subplots(1, 2, figsize=(8.0, 6.2))
    fig_t, axes_t = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex="row")

    for c_idx, (key, res) in enumerate(results.items()):
        dr, case, paper = res["drape"], res["case"], PAPER[key]
        plan = dr["xyz"][:, :2]
        tri = mtri.Triangulation(plan[:, 0], plan[:, 1], dr["tris"])
        tdry = 10.0 * case["tmax"]
        fnode = {v: nodal_fill_time(dr["tris"], len(plan), ft, tdry)
                 for v, ft in res["ft"].items()}
        y_line, x_line = symmetry_lines(dr)
        outline = np.array(paper["outline"], float)

        # ---- drape / shear map ----
        ax = axes_d[c_idx]
        tpc = ax.tripcolor(tri, facecolors=np.abs(dr["shear"]), cmap="Blues",
                           vmin=0, vmax=40, edgecolors="none")
        ax.plot(outline[:, 0], outline[:, 1], color=INK2, lw=1.0,
                label="paper's draped ply outline")
        ax.add_patch(plt.Circle((0, 0), INLET_RADIUS, color=ORANGE, lw=0))
        style_plan_axis(ax)
        ax.grid(False)
        ax.set_title(f"{case['label']}\n|shear| max {np.abs(dr['shear']).max():.0f}°",
                     fontsize=9, color=INK2, loc="left")
        if c_idx == 0:
            ax.set_ylabel("distance from inlet, y [mm]")
            ax.legend(loc="upper left", fontsize=8)
        print(f"  {key}: draped ply edge on x axis at "
              f"{plan[x_line, 0].max():.0f} mm (paper {outline[-1, 0]:.0f}), "
              f"far end on y axis at {plan[y_line, 1].max():.0f} mm "
              f"(paper {outline[0, 1]:.0f})")

        # ---- fronts at the measured times ----
        times = paper["times"]
        fig_f, axes_f = plt.subplots(1, len(times), figsize=(11.5, 4.6),
                                     sharey=True)
        for k, t in enumerate(times):
            ax = axes_f[k]
            ax.plot(outline[:, 0], outline[:, 1], color=AXIS, lw=1.0)
            segs = {v: front_contour(tri, fnode[v], t) for v in fnode}
            for s in segs["iso"]:
                ax.plot(s[:, 0], s[:, 1], color=ORANGE, lw=2.0)
            for s in segs["shear"]:
                ax.plot(s[:, 0], s[:, 1], color=BLUE, lw=2.0)
            for s in segs.get("shear_warp", []):
                ax.plot(s[:, 0], s[:, 1], color=AQUA, lw=2.0)
            for name, color in (("basic", ORANGE), ("model", BLUE)):
                c = np.array(paper[name][t], float)
                ax.plot(c[:, 0], c[:, 1], color=color, lw=1.2, ls=(0, (4, 2)))
            c = np.array(paper["exp"][t], float)
            ax.plot(c[:, 0], c[:, 1], color=INK, lw=2.0)
            style_plan_axis(ax)
            ax.set_title(f"t = {t} s", fontsize=9, color=INK2, loc="left")
            if k == 0:
                ax.set_ylabel("distance from inlet, y [mm]")

            # Errors: radial distance to the experimental front.
            r_exp = polar_profile([np.array(paper["exp"][t], float)], theta)
            mae = {}
            ours = [(f"ours_{v}", polar_profile(segs[v], theta)) for v in segs]
            for name, prof in (
                    *ours,
                    ("paper_model", polar_profile([np.array(paper["model"][t], float)], theta)),
                    ("paper_basic", polar_profile([np.array(paper["basic"][t], float)], theta))):
                mae[name] = float(np.nanmean(np.abs(prof - r_exp)))
            r_pb = polar_profile([np.array(paper["basic"][t], float)], theta)
            mae["iso_vs_paper_basic"] = float(np.nanmean(np.abs(
                polar_profile(segs["iso"], theta) - r_pb)))

            ey, ex = curve_axis_values(paper["exp"][t])
            row = dict(case=key, t=t, exp_y=ey, exp_x=ex)
            for v in fnode:
                row[f"{v}_y"] = float(axis_front(plan, y_line, fnode[v], 1, [t])[0])
                row[f"{v}_x"] = float(axis_front(plan, x_line, fnode[v], 0, [t])[0])
            for name in ("model", "basic"):
                py, px = curve_axis_values(paper[name][t])
                row[f"paper_{name}_y"], row[f"paper_{name}_x"] = py, px
            row.update({f"mae_{k_}": v_ for k_, v_ in mae.items()})
            rows.append(row)

        handles = [
            plt.Line2D([], [], color=INK, lw=2.0, label="experiment"),
            plt.Line2D([], [], color=BLUE, lw=2.0,
                       label="RTMsim-Py, shear-dependent K"),
            plt.Line2D([], [], color=BLUE, lw=1.2, ls=(0, (4, 2)),
                       label="paper (Fluent), shear-dependent K"),
            *([plt.Line2D([], [], color=AQUA, lw=2.0,
                          label="RTMsim-Py, K(γ), K1 from the warp")]
              if "shear_warp" in fnode else []),
            plt.Line2D([], [], color=ORANGE, lw=2.0,
                       label="RTMsim-Py, isotropic K"),
            plt.Line2D([], [], color=ORANGE, lw=1.2, ls=(0, (4, 2)),
                       label="paper (Fluent), isotropic K"),
            plt.Line2D([], [], color=AXIS, lw=1.0, label="paper's ply outline"),
        ]
        fig_f.legend(handles=handles, loc="lower center",
                     ncol=(len(handles) + 1) // 2 if len(handles) > 6 else 6,
                     fontsize=8)
        fig_f.suptitle(f"Flow fronts, {case['label']} sample (plan view, "
                       f"quarter model)", x=0.01, ha="left", fontsize=10)
        fig_f.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig_f.savefig(os.path.join(outdir, f"fronts_{key}.png"), dpi=150)
        plt.close(fig_f)

        # ---- front position vs time on both symmetry axes ----
        tt = np.linspace(0, case["tmax"], 400)
        for a_idx, (line, coord, name) in enumerate(
                ((y_line, 1, "long axis (y)"), (x_line, 0, "short axis (x)"))):
            ax = axes_t[c_idx, a_idx]
            for v, color in (("iso", ORANGE), ("shear", BLUE),
                             ("shear_warp", AQUA)):
                if v not in fnode:
                    continue
                ax.plot(tt, axis_front(plan, line, fnode[v], coord, tt),
                        color=color, lw=2.0)
            ax_key = "y" if coord == 1 else "x"
            for name_p, color, mk in (("model", BLUE, "s"),
                                      ("basic", ORANGE, "D")):
                vals = [curve_axis_values(paper[name_p][t])[1 - coord]
                        for t in times]
                ax.plot(times, vals, mk, ms=6, mfc=SURFACE, mec=color,
                        mew=1.4, ls="none")
            vals = [curve_axis_values(paper["exp"][t])[1 - coord] for t in times]
            ax.plot(times, vals, "o", ms=8, color=INK, mec=SURFACE, mew=1.0,
                    ls="none")
            if coord == 1:
                r = np.linspace(INLET_RADIUS + 1, 400, 300)
                ax.plot(radial_fill_time(r, case["mu"]), r, color=MUTED,
                        lw=1.0, ls=(0, (1, 2)))
            ax.set_xlim(0, case["tmax"])
            ax.set_ylim(0, 420 if coord == 1 else 260)
            ax.set_title(f"{case['label']}: front on the {name}",
                         fontsize=9, color=INK2, loc="left")
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"front position {ax_key} [mm]")

    handles = [
        plt.Line2D([], [], color=INK, marker="o", ms=7, ls="none",
                   label="experiment"),
        plt.Line2D([], [], color=BLUE, lw=2.0, label="RTMsim-Py, shear-dependent K"),
        plt.Line2D([], [], color=BLUE, marker="s", mfc=SURFACE, ls="none",
                   label="paper (Fluent), shear-dependent K"),
        *([plt.Line2D([], [], color=AQUA, lw=2.0,
                      label="RTMsim-Py, K(γ), K1 from the warp")]
          if any("shear_warp" in r["ft"] for r in results.values()) else []),
        plt.Line2D([], [], color=ORANGE, lw=2.0, label="RTMsim-Py, isotropic K"),
        plt.Line2D([], [], color=ORANGE, marker="D", mfc=SURFACE, ls="none",
                   label="paper (Fluent), isotropic K"),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (1, 2)),
                   label="flat radial Darcy flow, isotropic K"),
    ]
    fig_t.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig_t.tight_layout(rect=(0, 0.09, 1, 1))
    fig_t.savefig(os.path.join(outdir, "front_vs_time.png"), dpi=150)
    plt.close(fig_t)

    cb = fig_d.colorbar(tpc, ax=axes_d.tolist(), fraction=0.03, pad=0.02)
    cb.set_label("|shear angle| [deg]", color=INK2)
    cb.outline.set_visible(False)
    fig_d.suptitle("Kinematic drape on the double dome tool",
                   x=0.01, ha="left", fontsize=10)
    fig_d.savefig(os.path.join(outdir, "drape_shear.png"), dpi=150)
    plt.close(fig_d)

    print_summary(rows, results)
    return rows


def print_summary(rows, results):
    warp = "shear_warp_y" in rows[0]
    print("\n[3] Front positions on the symmetry axes [mm] (plan view)")
    hdr = (f"  {'case':5s} {'t[s]':>5s} | {'exp':>5s} {'ours K(γ)':>9s} "
           + (f"{'K1 warp':>8s} " if warp else "")
           + f"{'ours iso':>8s} {'Fluent K(γ)':>11s} {'Fluent iso':>10s}")
    for axis in ("y", "x"):
        print(f"  -- front on the {'long (y)' if axis == 'y' else 'short (x)'} axis")
        print(hdr)
        for r in rows:
            f = lambda v: "   edge" if not np.isfinite(v) else f"{v:7.0f}"
            print(f"  {r['case']:5s} {r['t']:5d} | {f(r['exp_'+axis]):>5s} "
                  f"{f(r['shear_'+axis]):>9s} "
                  + (f"{f(r['shear_warp_'+axis]):>8s} " if warp else "")
                  + f"{f(r['iso_'+axis]):>8s} "
                  f"{f(r['paper_model_'+axis]):>11s} "
                  f"{f(r['paper_basic_'+axis]):>10s}")
    print("\n  Mean radial distance to the experimental front [mm]")
    print(f"  {'case':5s} {'t[s]':>5s} | {'ours K(γ)':>9s} "
          + (f"{'K1 warp':>8s} " if warp else "")
          + f"{'ours iso':>8s} "
          f"{'Fluent K(γ)':>11s} {'Fluent iso':>10s} | ours iso vs Fluent iso")
    for r in rows:
        print(f"  {r['case']:5s} {r['t']:5d} | {r['mae_ours_shear']:9.1f} "
              + (f"{r['mae_ours_shear_warp']:8.1f} " if warp else "")
              + f"{r['mae_ours_iso']:8.1f} {r['mae_paper_model']:11.1f} "
              f"{r['mae_paper_basic']:10.1f} | {r['mae_iso_vs_paper_basic']:8.1f}")
    print("  Mean over the measured times: "
          + ", ".join(f"{name} {np.mean([r['mae_' + name] for r in rows]):.1f}"
                      for name in (["ours_shear"]
                                   + (["ours_shear_warp"] if warp else [])
                                   + ["ours_iso", "paper_model",
                                      "paper_basic"])))

    print("\n  Mean front speed on the long axis between measured times [mm/s]")
    for key in results:
        rr = [r for r in rows if r["case"] == key]
        for a, b in zip(rr[:-1], rr[1:]):
            dt = b["t"] - a["t"]
            sp = lambda k: (b[k] - a[k]) / dt
            print(f"  {key:5s} {a['t']:5d}-{b['t']:<5d} exp {sp('exp_y'):6.3f}  "
                  f"ours K(γ) {sp('shear_y'):6.3f}  "
                  + (f"K1 warp {sp('shear_warp_y'):6.3f}  " if warp else "")
                  + f"ours iso {sp('iso_y'):6.3f}  "
                  f"Fluent K(γ) {sp('paper_model_y'):6.3f}")


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in r.items()})


if __name__ == "__main__":
    main()
