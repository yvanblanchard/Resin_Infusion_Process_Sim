"""
RTMsim-Py test: annulus-filler shell mesh from a Nastran BDF.

Loads:
  assets/mesh_annulusfiller1.bdf      Nastran shell mesh (CTRIA3)
  assets/input_case3_coarsemesh.txt   solver parameters

The BDF carries 341 GRID nodes, 657 CTRIA3 triangles, and
`SET 1 = 535,536` marks the inlet elements. Solver is RTMsim
i_model=1 (compressible-air RTM) with the isotropic single-ply
preform described in the input file.
"""
import os
import re
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmsim as rtm


HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
BDF_PATH = os.path.join(ASSETS, "mesh_annulusfiller1.bdf")
INP_PATH = os.path.join(ASSETS, "input_case3_coarsemesh.txt")
OUTDIR = os.path.join(HERE, "output_annulusfiller")

# Cascade (delayed) injection port: location in mesh coordinates and
# the time at which it activates. The closest mesh cell to this point
# becomes a second inlet at t = CASCADE_T_ACTIVATE.
CASCADE_POINT = (0.10, 0.06, 0.24)
CASCADE_T_ACTIVATE = 200.0
CASCADE_RADIUS = 0.012   # use cells within this distance, fallback to closest 1


def parse_nastran_float(s):
    """Decode a Nastran 8-char float field, including '1.5-3' shorthand."""
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        for i in range(1, len(s)):
            if s[i] in "+-" and s[i - 1] not in "EeDd":
                s = s[:i] + "E" + s[i:]
                break
        return float(s.replace("D", "E").replace("d", "e"))


def read_bdf(path):
    """Return (node_id_to_xyz dict, list of (eid, [n1,n2,n3]), set of inlet eids)."""
    nodes = {}
    tris = []
    set_lines = {}
    current_set = None
    in_bulk = False
    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()
            if not stripped or stripped.startswith("$"):
                continue
            if stripped.upper().startswith("BEGIN BULK"):
                in_bulk = True
                continue
            if stripped.upper().startswith("ENDDATA"):
                break
            upper = stripped.upper()
            if upper.startswith("SET"):
                m = re.match(r"SET\s+(\d+)\s*=\s*(.*)", stripped, re.IGNORECASE)
                if m:
                    current_set = int(m.group(1))
                    set_lines[current_set] = m.group(2)
                continue
            if in_bulk:
                card = line[:8].strip().upper()
                if card == "GRID":
                    nid = int(line[8:16])
                    x = parse_nastran_float(line[24:32])
                    y = parse_nastran_float(line[32:40])
                    z = parse_nastran_float(line[40:48])
                    nodes[nid] = (x, y, z)
                elif card == "CTRIA3":
                    eid = int(line[8:16])
                    n1 = int(line[24:32])
                    n2 = int(line[32:40])
                    n3 = int(line[40:48])
                    tris.append((eid, [n1, n2, n3]))
    inlet_eids = set()
    for sid, txt in set_lines.items():
        if sid == 1:
            for tok in re.split(r"[,\s]+", txt.strip()):
                if tok:
                    inlet_eids.add(int(tok))
    return nodes, tris, inlet_eids


def parse_input_file(path):
    """
    Parse the legacy RTMsim text input. Pull out only what the Python
    port needs: model id, run control, EOS / fluid, the patch-1 (default
    preform) ply, and patch types.
    """
    raw_lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if s:
                raw_lines.append(s)

    def fl(parts):
        return [float(x) for x in parts]

    p = {}
    p["i_model"] = int(fl(raw_lines[0].split())[0])
    # raw_lines[1] is meshfilename (ignored)
    p["tmax"] = float(raw_lines[2])
    pref, rho_ref, gamma, mu = fl(raw_lines[3].split())
    p.update(p_ref=pref, rho_ref=rho_ref, gamma_eos=gamma, mu_resin=mu)
    p_a, p_init = fl(raw_lines[4].split())
    p.update(p_inlet=p_a, p_init=p_init)
    # default preform ply (line 5): t porosity K alpha refdir1 refdir2 refdir3
    t, phi, K, alpha, r1, r2, r3 = fl(raw_lines[5].split())
    p["preform"] = dict(thickness=t, porosity=phi, K1=K, K2=K * alpha,
                        refdir=(r1, r2, r3))
    # patch types are 5 lines further down (4 ply lines then patchtypes)
    # raw_lines[6..9] are patch1..patch4 ply (we ignore -- only one stack here)
    pt = [int(x) for x in raw_lines[10].split()]
    p["patch_types"] = pt + [0] * (4 - len(pt))
    p["n_pics"] = int(raw_lines[13]) if len(raw_lines) > 13 else 16
    return p


def build_mesh_from_bdf(path):
    nodes_dict, tris, inlet_eids = read_bdf(path)
    nid_sorted = sorted(nodes_dict.keys())
    nid_to_idx = {nid: k for k, nid in enumerate(nid_sorted)}
    nodes_arr = np.array([nodes_dict[nid] for nid in nid_sorted], dtype=np.float64)

    tri_arr = np.zeros((len(tris), 3), dtype=np.int64)
    eid_to_cellidx = {}
    for k, (eid, ns) in enumerate(tris):
        tri_arr[k] = [nid_to_idx[n] for n in ns]
        eid_to_cellidx[eid] = k

    inlet_cells = np.array(
        sorted(eid_to_cellidx[e] for e in inlet_eids if e in eid_to_cellidx),
        dtype=int,
    )
    mesh = rtm.build_mesh(nodes_arr, tri_arr,
                          patch_cell_ids=[inlet_cells, [], [], []])
    return mesh, nodes_arr, tri_arr, inlet_cells


def find_cells_near_point(mesh, point, radius):
    """Return cell indices whose cellcenter lies within `radius` of point.
    If none are within radius, fall back to the single closest cell."""
    p = np.asarray(point, dtype=np.float64)
    d = np.linalg.norm(mesh.cellcenter - p, axis=1)
    within = np.where(d <= radius)[0]
    if within.size:
        return within.astype(int), float(d[within].min())
    closest = int(np.argmin(d))
    return np.array([closest], dtype=int), float(d[closest])


def build_params(p_in, n_pics_override=None, cascade_events=None):
    pf = p_in["preform"]
    ply = rtm.PlyProperties(
        thickness=pf["thickness"], porosity=pf["porosity"],
        K1=pf["K1"], K2=pf["K2"],
        refdir=np.asarray(pf["refdir"], dtype=np.float64),
    )
    stack = rtm.LaminateStack(plies=[ply])
    n_pics = n_pics_override if n_pics_override is not None else p_in["n_pics"]
    params = rtm.SimParameters(
        i_model=p_in["i_model"],
        tmax=p_in["tmax"],
        p_ref=p_in["p_ref"], rho_ref=p_in["rho_ref"], gamma=p_in["gamma_eos"],
        mu_resin=p_in["mu_resin"],
        p_inlet=p_in["p_inlet"], p_init=p_in["p_init"],
        patch_types=[
            rtm.PATCH_INLET if t == 1 else
            rtm.PATCH_OUTLET if t == 3 else
            rtm.PATCH_IGNORE
            for t in p_in["patch_types"]
        ],
        n_pics=n_pics,
        stack=stack,
        cascade_events=list(cascade_events or []),
    )
    return params


# ---------- plotting ----------
def plot_mesh_overview(nodes, tris, inlet_cells, outpath,
                       cascade_cells=None, cascade_point=None,
                       cascade_t_activate=None):
    """3D wire/face view of the shell + highlighted inlet (and cascade) elements."""
    cascade_cells = np.asarray([] if cascade_cells is None else cascade_cells,
                               dtype=int)
    fig = plt.figure(figsize=(11, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    verts = nodes[tris]
    base = Poly3DCollection(verts, facecolor=(0.85, 0.88, 0.95),
                            edgecolor=(0.2, 0.2, 0.2), linewidths=0.2,
                            alpha=0.9)
    ax3d.add_collection3d(base)
    if len(inlet_cells):
        inlet_verts = nodes[tris[inlet_cells]]
        inlet_poly = Poly3DCollection(inlet_verts, facecolor="red",
                                      edgecolor="black", linewidths=0.6)
        ax3d.add_collection3d(inlet_poly)
    if cascade_cells.size:
        casc_verts = nodes[tris[cascade_cells]]
        casc_poly = Poly3DCollection(casc_verts, facecolor="orange",
                                     edgecolor="black", linewidths=0.6)
        ax3d.add_collection3d(casc_poly)
    if cascade_point is not None:
        cp = np.asarray(cascade_point)
        ax3d.scatter([cp[0]], [cp[1]], [cp[2]], c="orange",
                     edgecolor="black", s=70, zorder=10)
    mn = nodes.min(axis=0)
    mx = nodes.max(axis=0)
    ctr = 0.5 * (mn + mx)
    half = 0.55 * (mx - mn).max()
    ax3d.set_xlim(ctr[0] - half, ctr[0] + half)
    ax3d.set_ylim(ctr[1] - half, ctr[1] + half)
    ax3d.set_zlim(ctr[2] - half, ctr[2] + half)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    title = (f"Annulus filler mesh — {len(tris)} cells\n"
             f"red = primary inlet ({len(inlet_cells)} cells)")
    if cascade_cells.size:
        title += (f", orange = cascade ({cascade_cells.size} cells, "
                  f"t={cascade_t_activate:.0f}s)")
    ax3d.set_title(title)

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.triplot(nodes[:, 0], nodes[:, 2], tris, color="0.3", lw=0.3)
    if len(inlet_cells):
        ic = tris[inlet_cells]
        cx = nodes[ic, 0].mean(axis=1)
        cz = nodes[ic, 2].mean(axis=1)
        ax2d.scatter(cx, cz, c="red", s=30, zorder=5, label="primary inlet")
    if cascade_cells.size:
        cc_tris = tris[cascade_cells]
        cx = nodes[cc_tris, 0].mean(axis=1)
        cz = nodes[cc_tris, 2].mean(axis=1)
        ax2d.scatter(cx, cz, c="orange", s=40, zorder=5,
                     edgecolor="black",
                     label=f"cascade @ t={cascade_t_activate:.0f}s")
    if cascade_point is not None:
        cp = np.asarray(cascade_point)
        ax2d.scatter([cp[0]], [cp[2]], marker="x", c="black", s=60,
                     zorder=6, label=f"target {tuple(cp)}")
    ax2d.legend(loc="upper right", fontsize=8)
    ax2d.set_xlabel("x [m]")
    ax2d.set_ylabel("z [m]")
    ax2d.set_aspect("equal")
    ax2d.set_title("(x, z) projection")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _gamma_with_inlet(snap):
    g = snap.gamma.copy()
    g[snap.celltype == -1] = 1.0
    return g


def plot_fill_3d_panel(ax, nodes, tris, snap, cmap="RdYlBu_r",
                       norm=None, title=None):
    g = _gamma_with_inlet(snap)
    verts = nodes[tris]
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=1.0)
    cm = plt.get_cmap(cmap)
    facecolors = cm(norm(g))
    coll = Poly3DCollection(verts, facecolor=facecolors,
                            edgecolor=(0.1, 0.1, 0.1), linewidths=0.05)
    ax.add_collection3d(coll)
    mn = nodes.min(axis=0)
    mx = nodes.max(axis=0)
    ctr = 0.5 * (mn + mx)
    half = 0.55 * (mx - mn).max()
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    interior = snap.celltype != -1
    mean_g = snap.gamma[interior].mean()
    base_title = f"t = {snap.t:.1f} s   mean γ = {mean_g:.2f}"
    ax.set_title(base_title if title is None else f"{title}\n{base_title}",
                 fontsize=9)
    return coll


def plot_fill_2d_panel(ax, nodes, tris, snap, cmap="RdYlBu_r", norm=None,
                       title=None):
    g = _gamma_with_inlet(snap)
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=1.0)
    tpc = ax.tripcolor(nodes[:, 0], nodes[:, 2], tris, facecolors=g,
                       cmap=cmap, norm=norm, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    interior = snap.celltype != -1
    mean_g = snap.gamma[interior].mean()
    base_title = f"t = {snap.t:.1f} s   mean γ = {mean_g:.2f}"
    ax.set_title(base_title if title is None else f"{title}\n{base_title}",
                 fontsize=9)
    return tpc


def plot_sequence_2d(nodes, tris, snaps, outpath, suptitle):
    n_show = min(8, len(snaps))
    indices = np.linspace(0, len(snaps) - 1, n_show, dtype=int)
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows + 0.4))
    axes = np.atleast_2d(axes)
    norm = Normalize(vmin=0.0, vmax=1.0)
    last = None
    flat = list(axes.flat)
    for k, idx in enumerate(indices):
        last = plot_fill_2d_panel(flat[k], nodes, tris, snaps[idx],
                                  norm=norm, title=f"snap {idx}")
    for ax in flat[n_show:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    if last is not None:
        sm = ScalarMappable(norm=norm, cmap="RdYlBu_r")
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.03)
        cbar.set_label("filling fraction γ")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_3d(nodes, tris, snaps, outpath, suptitle):
    pick = [0.10, 0.30, 0.55, 0.85]
    interior_mask_template = None
    chosen_idx = []
    for tgt in pick:
        # pick snap with mean gamma closest to target
        best = 0
        best_diff = 1e9
        for i, s in enumerate(snaps):
            interior = s.celltype != -1
            g = s.gamma[interior].mean()
            d = abs(g - tgt)
            if d < best_diff:
                best_diff = d
                best = i
        chosen_idx.append(best)
    n = len(chosen_idx)
    fig = plt.figure(figsize=(4.5 * n, 4.2))
    norm = Normalize(vmin=0.0, vmax=1.0)
    coll_last = None
    for k, idx in enumerate(chosen_idx):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        coll_last = plot_fill_3d_panel(ax, nodes, tris, snaps[idx],
                                       norm=norm, title=f"snap {idx}")
    fig.suptitle(suptitle, fontsize=12)
    sm = ScalarMappable(norm=norm, cmap="RdYlBu_r")
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.18, 0.012, 0.65])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("filling fraction γ")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_fill_curve(snaps, outpath, cascade_t_activate=None):
    ts = [s.t for s in snaps]
    means = []
    for s in snaps:
        interior = s.celltype != -1
        means.append(s.gamma[interior].mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, means, "o-", color="C0", lw=1.5)
    if cascade_t_activate is not None:
        ax.axvline(cascade_t_activate, color="orange", ls="--", lw=1.5,
                   label=f"cascade activates @ t={cascade_t_activate:.0f}s")
        ax.legend(loc="lower right")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean filling fraction γ (interior cells)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("Annulus filler — fill progress")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"Reading mesh: {BDF_PATH}")
    mesh, nodes, tris, inlet_cells = build_mesh_from_bdf(BDF_PATH)
    print(f"  -> {mesh.N} cells, {nodes.shape[0]} nodes, "
          f"{len(inlet_cells)} inlet cells")

    print(f"Reading input: {INP_PATH}")
    p_in = parse_input_file(INP_PATH)
    print(f"  i_model={p_in['i_model']}, tmax={p_in['tmax']} s, "
          f"p_inlet={p_in['p_inlet']:.3e} Pa, p_init={p_in['p_init']:.3e} Pa")
    pf = p_in["preform"]
    print(f"  ply: t={pf['thickness']*1e3:.2f}mm, phi={pf['porosity']}, "
          f"K1={pf['K1']:.2e}, K2={pf['K2']:.2e}, refdir={pf['refdir']}")

    cascade_cells, cascade_dist = find_cells_near_point(
        mesh, CASCADE_POINT, CASCADE_RADIUS)
    print(f"\nCascade injection at {CASCADE_POINT}:")
    print(f"  -> {cascade_cells.size} cell(s) within {CASCADE_RADIUS} m "
          f"(closest at {cascade_dist*1e3:.2f} mm)")
    print(f"  -> activates at t = {CASCADE_T_ACTIVATE:.1f} s")

    plot_mesh_overview(nodes, tris, inlet_cells,
                       os.path.join(OUTDIR, "mesh_overview.png"),
                       cascade_cells=cascade_cells,
                       cascade_point=CASCADE_POINT,
                       cascade_t_activate=CASCADE_T_ACTIVATE)

    params = build_params(
        p_in,
        cascade_events=[(CASCADE_T_ACTIVATE, cascade_cells)],
    )
    print("\nRunning solver (first call may JIT-compile)...")
    t0 = time.time()
    snaps = rtm.run_filling(mesh, params)
    print(f"  -> {time.time() - t0:.1f}s, {len(snaps)} snapshots")

    final = snaps[-1]
    interior = final.celltype != -1
    print(f"\nFinal mean gamma = {final.gamma[interior].mean():.3f} "
          f"at t = {final.t:.2f} s")

    plot_sequence_2d(nodes, tris, snaps,
                     os.path.join(OUTDIR, "fill_sequence_2d.png"),
                     "Annulus filler fill — (x, z) projection")
    plot_sequence_3d(nodes, tris, snaps,
                     os.path.join(OUTDIR, "fill_sequence_3d.png"),
                     "Annulus filler fill — 3D snapshots")
    plot_fill_curve(snaps, os.path.join(OUTDIR, "fill_progress.png"),
                    cascade_t_activate=CASCADE_T_ACTIVATE)

    print(f"\nFigures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
