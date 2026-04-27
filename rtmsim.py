"""
RTMsim-Py: Python port of RTMsim/LCMsim (Obertscheider et al., FHWN).

Finite Area Method solver for LCM filling simulation on triangle shell
meshes. Three process models are supported:

    i_model=1  RTM (resin transfer molding), iso-thermal compressible
               Euler + Darcy drag + smooth VOF, quadratic-fit EOS.
    i_model=2  RTM-VARI two-fluid surrogate, hard VOF cutoff, power-
               law EOS p(rho) mapping rho_air -> p_init, rho_resin
               -> p_inlet.
    i_model=3  VARI without flow distribution medium: same EOS as
               model 2, with pressure-dependent porosity (preform
               compaction). Per-ply quadratic phi(p) = phi0 + c*p^2;
               permeability scales by Carman-Kozeny-like factor
               phi^3/(1-phi)^2.

Meshes are built from numpy arrays. Per-element permeability comes
from a `LaminateStack` (PCOMP-style: every cell carries the same set
of plies). Patches mark inlet/outlet regions only; the stack supplies
all material/orientation data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PATCH_IGNORE = 0
PATCH_INLET = 1
PATCH_PREFORM = 2
PATCH_OUTLET = 3
CELL_INTERIOR = 1
CELL_INLET = -1
CELL_OUTLET = -2
CELL_WALL = -3


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class PatchProperties:
    """
    Marker for an inlet/outlet patch. Carries only a thickness fallback
    used when scaling boundary face areas. Fibre orientation, in-plane
    permeability, and porosity are now owned exclusively by the stack.
    """
    thickness: float = 3e-3


@dataclass
class PlyProperties:
    """
    One ply in a stacked-laminate (PCOMP) element.

    Per-ply: thickness, in-plane principal permeabilities (K1 along the
    fibre, K2 across), global-frame fibre direction, baseline porosity.

    For i_model=3 (VARI compactable preform) two extra parameters define
    the quadratic porosity-vs-pressure law:

        phi(p) = porosity + c * p^2,
        c      = (porosity_at_p1 - porosity) / p1^2

    porosity_at_p1 should be the measured / specified porosity at
    pressure p1 (typically the injection pressure). The default makes
    the law a no-op (porosity_at_p1 == porosity), so non-LCM users get
    constant porosity.
    """
    thickness: float = 0.75e-3
    porosity: float = 0.7
    K1: float = 3e-10
    K2: float = 0.6e-10
    refdir: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    porosity_at_p1: float = 0.7
    p1: float = 1.0e5

    @property
    def porosity_quadratic_c(self) -> float:
        if self.p1 <= 0:
            return 0.0
        return (self.porosity_at_p1 - self.porosity) / (self.p1 * self.p1)


@dataclass
class LaminateStack:
    """
    Stack of plies covering the full mesh — one stack shared by every
    element. The element's geometric (e1, e2, e3) frame is built from
    its triangle nodes; each ply's `refdir` is projected into the
    element's tangent plane to get a local angle, then the ply's
    diagonal local-frame tensor diag(K1, K2) is rotated into the
    element frame and summed thickness-weighted across plies. The
    result is a single 2x2 tensor per element.

    Standard LCM PCOMP shell assumption: same in-plane pressure
    gradient across all plies (parallel flow), so flow rates add —
    equivalent to thickness-weighted tensor averaging.

    For i_model=3 the quadratic porosity coefficient c is also linear
    in stack averaging, so per-element phi(p) is a single quadratic
    phi_eff(p) = phi0_eff + c_eff * p^2.
    """
    plies: list = field(default_factory=list)

    @property
    def total_thickness(self) -> float:
        return float(sum(p.thickness for p in self.plies))

    @property
    def effective_porosity(self) -> float:
        t = self.total_thickness
        if t == 0:
            return 0.0
        return float(sum(p.porosity * p.thickness for p in self.plies) / t)

    @property
    def effective_c_porosity(self) -> float:
        """Thickness-weighted c-coefficient for phi_eff(p) = phi0 + c*p^2."""
        t = self.total_thickness
        if t == 0:
            return 0.0
        return float(sum(p.porosity_quadratic_c * p.thickness
                         for p in self.plies) / t)


@dataclass
class SimParameters:
    # --- model selection ---
    i_model: int = 1                 # 1=RTM, 2=RTM-VARI, 3=VARI compactable

    # --- run control ---
    tmax: float = 200.0
    n_pics: int = 16
    max_neighbours: int = 10
    gradient_method: int = 3
    cfl: float = 0.05

    # --- fluid / EOS (i_model=1) ---
    p_ref: float = 1.01325e5
    rho_ref: float = 1.225
    gamma: float = 1.4
    mu_resin: float = 0.06
    p_inlet: float = 1.35e5
    p_init: float = 1.0e5

    # --- two-fluid EOS (i_model=2,3) ---
    rho_air: float = 1.225
    rho_resin: float = 960.0
    # Power-law exponent in p(rho). 4 for normal preforms; auto-bumped
    # to 25 when the in-plane permeability ratio exceeds 100
    # (race-tracking suppression). Set to a positive int to override.
    exp_eos: int = 0                 # 0 == auto

    # --- patches (inlet / outlet markers only) ---
    patches: List[PatchProperties] = field(
        default_factory=lambda: [PatchProperties() for _ in range(4)])
    patch_types: List[int] = field(
        default_factory=lambda: [PATCH_INLET, PATCH_IGNORE, PATCH_IGNORE, PATCH_IGNORE])

    # --- preform (required) ---
    # Stack must be set; supplies per-element K, porosity, and thickness.
    stack: object = None

    # --- cascade injection ---
    # Each event is a (t_activate, cell_ids) tuple. At simulation time
    # t_activate the listed cells are flipped to CELL_INLET and pinned at
    # inlet rho/p/gamma for the rest of the run, modelling a delayed
    # secondary injection port.
    cascade_events: list = field(default_factory=list)

    # ----------------------------------------------------------------------
    # Thermal + cure (1D through-thickness lumped model)
    # ----------------------------------------------------------------------
    # When thermal_enabled is False the solver behaves exactly as before.
    # When True, each cell carries a single temperature T (through-
    # thickness average); heat loss to the tool is modelled as a Newton
    # cooling sink 2*h_tool/h_thk*(T - T_tool) — the analytical 1D
    # through-thickness profile collapsed into a per-cell sink term.
    thermal_enabled: bool = False
    cure_enabled: bool = False

    # Initial / boundary conditions [K]
    T_init: float = 298.15        # initial preform/mold temperature
    T_inlet: float = 333.15       # incoming resin temperature
    T_tool: float = 333.15        # tool/wall temperature (Dirichlet sink)
    h_tool: float = 1000.0        # tool-resin heat transfer coeff [W/(m^2 K)]

    # Thermal properties
    cp_resin: float = 1800.0      # [J/(kg K)]
    cp_fiber: float = 800.0       # [J/(kg K)]
    rho_fiber: float = 2540.0     # [kg/m^3]
    k_resin: float = 0.2          # [W/(m K)]
    k_fiber: float = 1.0          # [W/(m K)]
    enable_inplane_conduction: bool = False  # usually negligible in shells

    # Cure exotherm (Kamal-Sourour)
    H_total: float = 350.0e3      # total heat of reaction [J/kg resin]
    A1_cure: float = 2.0e3        # [1/s]
    A2_cure: float = 2.0e5        # [1/s]
    E1_cure: float = 5.0e4        # [J/mol]
    E2_cure: float = 6.0e4        # [J/mol]
    m_cure: float = 0.5
    n_cure: float = 1.5
    alpha_init: float = 0.0       # initial conversion

    # Viscosity model: mu(T, alpha) = mu_inf * exp(E_mu/RT)
    #                                  * (alpha_g/(alpha_g - alpha))^(C1 + C2*alpha)
    # When cure_enabled is False the Castro factor is omitted.
    # Defaults are tuned so mu(T=333 K, alpha=0) ≈ 0.10 Pa s, matching
    # the existing iso-thermal demos. Real resins should override these.
    mu_inf: float = 2.0e-6        # [Pa s]
    E_mu: float = 3.0e4           # [J/mol]
    alpha_gel: float = 0.6
    C1_visc: float = 1.5
    C2_visc: float = 1.0
    mu_max: float = 1.0e3         # cap to avoid blow-up near gel [Pa s]

    def validate(self):
        if self.i_model not in (1, 2, 3):
            raise ValueError("i_model must be 1, 2, or 3")
        if self.tmax <= 0:
            raise ValueError("tmax must be > 0")
        if self.p_inlet <= self.p_init:
            raise ValueError("p_inlet must be > p_init")
        if self.thermal_enabled:
            if self.T_init <= 0 or self.T_inlet <= 0 or self.T_tool <= 0:
                raise ValueError("Temperatures must be > 0 K (use Kelvin)")
            if self.h_tool < 0 or self.cp_resin <= 0 or self.cp_fiber <= 0:
                raise ValueError("h_tool, cp_resin, cp_fiber must be positive")
        if self.cure_enabled:
            if not self.thermal_enabled:
                raise ValueError("cure_enabled requires thermal_enabled=True")
            if not (0.0 <= self.alpha_init < self.alpha_gel <= 1.0):
                raise ValueError("Need 0 <= alpha_init < alpha_gel <= 1")
        if self.stack is None or not getattr(self.stack, "plies", None):
            raise ValueError("A LaminateStack with plies is required")
        if self.i_model == 3:
            # Sanity-check the porosity quadratic at the operating
            # pressure: if phi(p_inlet) > 0.9 the preform is essentially
            # decompacted, the Carman-Kozeny-like factor phi^3/(1-phi)^2
            # diverges (>= ~73 already at phi=0.9), and dt has to
            # collapse to keep the solver stable. Tell the user instead.
            for k, ply in enumerate(self.stack.plies):
                ap = ply.porosity
                cp = ply.porosity_quadratic_c
                phi_at_inlet = ap + cp * (self.p_inlet ** 2)
                if phi_at_inlet > 0.9:
                    raise ValueError(
                        f"i_model=3: ply {k} porosity at p_inlet="
                        f"{self.p_inlet:.0f} Pa is {phi_at_inlet:.3f} "
                        f"(> 0.9). Reduce porosity_at_p1 or increase p1.")


@dataclass
class MeshData:
    N: int
    nodes: np.ndarray
    cellgridid: np.ndarray
    cellcenter: np.ndarray
    patch_cell_ids: List[np.ndarray] = field(
        default_factory=lambda: [np.array([], dtype=int)] * 4)


@dataclass
class CellGeom:
    neighbours: np.ndarray
    celltype: np.ndarray
    volume: np.ndarray
    cc_to_cc_x: np.ndarray
    cc_to_cc_y: np.ndarray
    T11: np.ndarray
    T12: np.ndarray
    T21: np.ndarray
    T22: np.ndarray
    face_nx: np.ndarray
    face_ny: np.ndarray
    face_area: np.ndarray


@dataclass
class Snapshot:
    step: int
    t: float
    gamma: np.ndarray
    p: np.ndarray
    celltype: np.ndarray
    # i_model=3: per-cell instantaneous porosity / compacted thickness
    porosity: np.ndarray = None
    thickness: np.ndarray = None
    # Absolute-pressure offset baked in by the solver:
    #   p_absolute = p + p_offset
    # i_model=1 runs in normalized pressure (origin at p_eps), so this
    # offset is (p_init - p_eps); for i_model=2/3 the stored p is
    # already absolute and the offset is 0.
    p_offset: float = 0.0
    # Thermal/cure state — None when thermal/cure are disabled.
    T: np.ndarray = None          # [K] per cell
    alpha: np.ndarray = None      # cure conversion in [0, 1]
    mu: np.ndarray = None         # [Pa s] per cell (instantaneous)


def pressure_absolute(snap):
    """Per-cell absolute pressure (Pa) for any model."""
    return snap.p + snap.p_offset


def pressure_results(snaps):
    """
    Bundle of pressure analysis results across all snapshots.

    Returns a dict with:
      times     (n_snap,)            time of each snapshot [s]
      p         (n_snap, n_cells)    absolute pressure per cell [Pa]
      p_min     (n_snap,)            min absolute pressure (fluid cells)
      p_max     (n_snap,)            max absolute pressure (fluid cells)
      p_mean    (n_snap,)            unweighted mean over fluid cells
      celltype  (n_snap, n_cells)    cell-type tag per snapshot
    Fluid cells are interior + wall (excludes inlet/outlet so the
    imposed-pressure boundary doesn't dominate stats).
    """
    times = np.array([s.t for s in snaps], dtype=np.float64)
    p_stack = np.array([pressure_absolute(s) for s in snaps], dtype=np.float64)
    ct_stack = np.array([s.celltype for s in snaps])
    p_min = np.empty(len(snaps))
    p_max = np.empty(len(snaps))
    p_mean = np.empty(len(snaps))
    for k, s in enumerate(snaps):
        fluid = (s.celltype == CELL_INTERIOR) | (s.celltype == CELL_WALL)
        if not fluid.any():
            p_min[k] = p_max[k] = p_mean[k] = np.nan
            continue
        pa = pressure_absolute(s)[fluid]
        p_min[k] = pa.min()
        p_max[k] = pa.max()
        p_mean[k] = pa.mean()
    return dict(times=times, p=p_stack, celltype=ct_stack,
                p_min=p_min, p_max=p_max, p_mean=p_mean)


# --------------------------------------------------------------------------
# Mesh construction
# --------------------------------------------------------------------------
def build_mesh(nodes, triangles, patch_cell_ids=None) -> MeshData:
    """Build MeshData from raw arrays. Triangles sorted to (min, med, max)."""
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    tris = np.ascontiguousarray(triangles, dtype=np.int64)
    tris = np.sort(tris, axis=1)
    cellcenter = nodes[tris].mean(axis=1)
    if patch_cell_ids is None:
        patch_cell_ids = [np.array([], dtype=int) for _ in range(4)]
    else:
        patch_cell_ids = list(patch_cell_ids) + \
                         [np.array([], dtype=int)] * (4 - len(patch_cell_ids))
        patch_cell_ids = [np.asarray(p, dtype=int) for p in patch_cell_ids[:4]]
    return MeshData(
        N=tris.shape[0], nodes=nodes, cellgridid=tris,
        cellcenter=cellcenter, patch_cell_ids=patch_cell_ids,
    )


def make_square_plate(side=0.3, n_div=40, z=0.0) -> MeshData:
    """Structured triangle mesh of a square plate centered at origin."""
    if n_div < 2:
        raise ValueError("n_div must be >= 2")
    L = side / 2.0
    xs = np.linspace(-L, L, n_div + 1)
    ys = np.linspace(-L, L, n_div + 1)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.full_like(XX, z)
    nodes = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    def nid(i, j): return i * (n_div + 1) + j
    tris = []
    for i in range(n_div):
        for j in range(n_div):
            n00 = nid(i, j)
            n10 = nid(i, j + 1)
            n01 = nid(i + 1, j)
            n11 = nid(i + 1, j + 1)
            if (i + j) % 2 == 0:
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])
            else:
                tris.append([n00, n10, n01])
                tris.append([n10, n11, n01])
    return build_mesh(nodes, np.array(tris, dtype=np.int64))


def assign_patch_by_predicate(mesh, patch_idx, predicate):
    if not 0 <= patch_idx < 4:
        raise IndexError(patch_idx)
    mask = predicate(mesh.cellcenter)
    mesh.patch_cell_ids[patch_idx] = np.where(mask)[0].astype(int)


def assign_patch_by_disk(mesh, patch_idx, center_xy, radius):
    cx, cy = center_xy
    def pred(cc):
        dx = cc[:, 0] - cx
        dy = cc[:, 1] - cy
        return (dx * dx + dy * dy) <= radius * radius
    assign_patch_by_predicate(mesh, patch_idx, pred)


# --------------------------------------------------------------------------
# Topology + local coord systems (port of tools.jl)
# --------------------------------------------------------------------------
def create_faces(mesh, max_neighbours=10):
    """Edge adjacency. Returns (neighbours, celltype)."""
    N = mesh.N
    cg = mesh.cellgridid
    edge_map = {}
    for c in range(N):
        n0, n1, n2 = int(cg[c, 0]), int(cg[c, 1]), int(cg[c, 2])
        for a, b in ((n0, n1), (n1, n2), (n0, n2)):
            edge_map.setdefault((a, b), []).append(c)
    neighbours = np.full((N, max_neighbours), -9, dtype=np.int64)
    celltype = np.full(N, CELL_INTERIOR, dtype=np.int64)
    fill = np.zeros(N, dtype=np.int64)
    for _, cells in edge_map.items():
        if len(cells) == 1:
            celltype[cells[0]] = CELL_WALL
        else:
            for i, ci in enumerate(cells):
                for j, cj in enumerate(cells):
                    if i == j:
                        continue
                    if fill[ci] >= max_neighbours:
                        raise RuntimeError(f"cell {ci} too many neighbours")
                    if cj not in neighbours[ci, :fill[ci]]:
                        neighbours[ci, fill[ci]] = cj
                        fill[ci] += 1
    return neighbours, celltype


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero refdir")
    return v / n


def assign_parameters(mesh, params, celltype):
    """
    Tag inlet/outlet cells from patch ids; build per-cell scalar arrays.

    All preform properties (thickness, porosity, K) come from the stack.
    Patches only mark inlet (PATCH_INLET) and outlet (PATCH_OUTLET)
    cells. PATCH_PREFORM is accepted but treated as IGNORE — the stack
    already covers every element.
    """
    N = mesh.N
    ct = celltype.copy()
    for pi in range(4):
        cells = mesh.patch_cell_ids[pi]
        t = params.patch_types[pi]
        if t == PATCH_INLET:
            ct[cells] = CELL_INLET
        elif t == PATCH_OUTLET:
            ct[cells] = CELL_OUTLET

    stack = params.stack
    t_tot = stack.total_thickness
    phi_eff = stack.effective_porosity
    thickness = np.full(N, t_tot)
    porosity  = np.full(N, phi_eff)
    viscosity = np.full(N, params.mu_resin)
    return ct, thickness, porosity, viscosity


def create_coordinate_systems(mesh, neighbours, celltype, thickness, max_neighbours):
    """Per-cell geometric basis from triangle nodes; flattened neighbour geometry.

    Local frame is purely geometric (no fibre rotation): e1 along the
    first triangle edge, e2 perpendicular in-plane, e3 = e1 x e2. The
    per-element K tensor (built by build_stack_tensor) is expressed in
    this frame, so no rotation by a single per-cell theta is needed.
    """
    N = mesh.N
    K = max_neighbours
    nodes = mesh.nodes
    cg = mesh.cellgridid
    cc = mesh.cellcenter
    b1 = np.zeros((N, 3))
    b2 = np.zeros((N, 3))
    b3 = np.zeros((N, 3))
    grid_local = np.zeros((N, 3, 3))
    theta = np.zeros(N)

    for ind in range(N):
        i1, i2, i3 = cg[ind]
        e1 = nodes[i2] - nodes[i1]
        e1 /= np.linalg.norm(e1)
        a2 = nodes[i3] - nodes[i1]
        a2 /= np.linalg.norm(a2)
        e2 = a2 - np.dot(e1, a2) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        th = 0.0
        theta[ind] = th
        c_th, s_th = np.cos(th), np.sin(th)
        b1[ind] =  c_th * e1 + s_th * e2
        b2[ind] = -s_th * e1 + c_th * e2
        b3[ind] = e3
        T = np.column_stack([b1[ind], b2[ind], b3[ind]])
        Tinv = np.linalg.inv(T)
        for kk, ivtx in enumerate((i1, i2, i3)):
            grid_local[ind, kk] = Tinv @ (nodes[ivtx] - cc[ind])

    FILL = -9.0
    volume = np.zeros(N)
    cc_to_cc_x = np.full((N, K), FILL)
    cc_to_cc_y = np.full((N, K), FILL)
    T11 = np.full((N, K), FILL)
    T12 = np.full((N, K), FILL)
    T21 = np.full((N, K), FILL)
    T22 = np.full((N, K), FILL)
    face_nx = np.full((N, K), FILL)
    face_ny = np.full((N, K), FILL)
    face_area = np.full((N, K), FILL)

    for ind in range(N):
        nbrs = neighbours[ind]
        nbrs = nbrs[nbrs >= 0]
        for k, nb in enumerate(nbrs):
            host_ids = set(cg[ind].tolist())
            nb_ids   = set(cg[nb].tolist())
            shared = sorted(host_ids & nb_ids)
            if len(shared) != 2:
                continue
            ga, gb = shared
            ka = int(np.where(cg[ind] == ga)[0][0])
            kb = int(np.where(cg[ind] == gb)[0][0])
            x0 = grid_local[ind, ka]
            r0 = grid_local[ind, kb] - grid_local[ind, ka]
            P = np.zeros(3)
            lam = np.dot(P - x0, r0) / np.dot(r0, r0)
            Q1 = x0 + lam * r0
            l1 = np.linalg.norm(P - Q1)
            if l1 < 1e-30:
                continue
            nvec = (Q1 - P) / l1
            face_nx[ind, k] = nvec[0]
            face_ny[ind, k] = nvec[1]

            T_host = np.column_stack([b1[ind], b2[ind], b3[ind]])
            A_local = np.linalg.solve(T_host, cc[nb] - cc[ind])
            lam_A = np.dot(A_local - x0, r0) / np.dot(r0, r0)
            Q2 = x0 + lam_A * r0
            l2 = np.linalg.norm(A_local - Q2)
            flat = P + (Q1 - P) + (Q2 - Q1) + (l2 / l1) * (Q1 - P)
            cc_to_cc_x[ind, k] = flat[0]
            cc_to_cc_y[ind, k] = flat[1]

            edge_len = np.linalg.norm(nodes[gb] - nodes[ga])
            face_area[ind, k] = 0.5 * (thickness[ind] + thickness[nb]) * edge_len

            nb_tri = cg[nb]
            other_mask = ~np.isin(nb_tri, list(shared))
            i_other = int(nb_tri[other_mask][0])
            A_other = np.linalg.solve(T_host, nodes[i_other] - cc[ind])
            lam_o = np.dot(A_other - x0, r0) / np.dot(r0, r0)
            Q3 = x0 + lam_o * r0
            l3 = np.linalg.norm(A_other - Q3)
            third_flat = P + (Q1 - P) + (Q3 - Q1) + (l3 / l1) * (Q1 - P)

            vtx = np.zeros((3, 3))
            for j, gid in enumerate(nb_tri):
                if gid == ga:
                    vtx[j] = grid_local[ind, ka]
                elif gid == gb:
                    vtx[j] = grid_local[ind, kb]
                else:
                    vtx[j] = third_flat
            f1 = vtx[1] - vtx[0]
            f1 /= np.linalg.norm(f1)
            a2 = vtx[2] - vtx[0]
            a2 /= np.linalg.norm(a2)
            f2 = a2 - np.dot(f1, a2) * f1
            f2 /= np.linalg.norm(f2)
            f3 = np.cross(f1, f2)
            th_nb = theta[nb]
            c_th, s_th = np.cos(th_nb), np.sin(th_nb)
            c1 =  c_th * f1 + s_th * f2
            c2 = -s_th * f1 + c_th * f2
            Tmat = np.column_stack([c1, c2, f3])
            T11[ind, k] = Tmat[0, 0]
            T12[ind, k] = Tmat[0, 1]
            T21[ind, k] = Tmat[1, 0]
            T22[ind, k] = Tmat[1, 1]

        v1 = grid_local[ind, 1] - grid_local[ind, 0]
        v2 = grid_local[ind, 2] - grid_local[ind, 0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        volume[ind] = thickness[ind] * area

    return CellGeom(
        neighbours=neighbours, celltype=celltype, volume=volume,
        cc_to_cc_x=cc_to_cc_x, cc_to_cc_y=cc_to_cc_y,
        T11=T11, T12=T12, T21=T21, T22=T22,
        face_nx=face_nx, face_ny=face_ny, face_area=face_area,
    )

def build_stack_tensor(mesh, stack):
    """
    Per-element 2x2 in-plane permeability tensor for a stacked laminate.

    For each element, build the local geometric frame (e1, e2, e3)
    from the triangle nodes. For each ply, project its global refdir
    into that element's tangent plane to get the angle in (e1, e2).
    The ply's tensor in the element frame is

        R(alpha_p) . diag(K1_p, K2_p) . R(alpha_p)^T

    The element-effective tensor is the thickness-weighted sum of ply
    tensors divided by total stack thickness. Porosity is constant per
    ply, so phi0_eff and the i_model=3 porosity quadratic coefficient
    c_eff are also stack averages of per-ply values (independent of
    element orientation). All four arrays are returned per element.
    """
    N = mesh.N
    nodes = mesh.nodes
    cg = mesh.cellgridid
    Kxx = np.zeros(N)
    Kxy = np.zeros(N)
    Kyy = np.zeros(N)
    t_tot = stack.total_thickness
    if t_tot <= 0:
        raise ValueError("Stack has zero total thickness")

    phi0_eff = stack.effective_porosity
    c_eff = stack.effective_c_porosity
    phi0 = np.full(N, phi0_eff)
    c_porosity = np.full(N, c_eff)

    for ind in range(N):
        i1, i2, i3 = cg[ind]
        e1 = nodes[i2] - nodes[i1]
        e1 /= np.linalg.norm(e1)
        a2 = nodes[i3] - nodes[i1]
        a2 /= np.linalg.norm(a2)
        e2 = a2 - np.dot(e1, a2) * e1
        e2 /= np.linalg.norm(e2)

        Kxx_e = 0.0
        Kxy_e = 0.0
        Kyy_e = 0.0
        for ply in stack.plies:
            r = np.asarray(ply.refdir, dtype=np.float64)
            rn = np.linalg.norm(r)
            if rn == 0:
                continue
            r = r / rn
            rx = float(r @ e1)
            ry = float(r @ e2)
            mag = (rx * rx + ry * ry) ** 0.5
            if mag < 1e-30:
                rx = 1.0
                ry = 0.0
            else:
                rx /= mag
                ry /= mag
            c = rx
            s = ry
            K1p = ply.K1
            K2p = ply.K2
            kxx = c * c * K1p + s * s * K2p
            kyy = s * s * K1p + c * c * K2p
            kxy = c * s * (K1p - K2p)
            w = ply.thickness
            Kxx_e += kxx * w
            Kxy_e += kxy * w
            Kyy_e += kyy * w

        Kxx[ind] = Kxx_e / t_tot
        Kxy[ind] = Kxy_e / t_tot
        Kyy[ind] = Kyy_e / t_tot

    return Kxx, Kxy, Kyy, phi0, c_porosity




# --------------------------------------------------------------------------
# Numerics (gradient + flux)
# --------------------------------------------------------------------------
def numerical_gradient(i_method, ind, p_old, neighbours, cc_to_cc_x, cc_to_cc_y):
    nbrs = neighbours[ind]
    nbrs = nbrs[nbrs >= 0]
    K = nbrs.size
    if K < 2:
        return 0.0, 0.0
    dx = cc_to_cc_x[ind, :K]
    dy = cc_to_cc_y[ind, :K]
    db = p_old[nbrs] - p_old[ind]
    if i_method == 3:
        a = float(dx @ dx)
        b_ = float(dx @ dy)
        d = float(dy @ dy)
        rhs_x = float(dx @ db)
        rhs_y = float(dy @ db)
        det = a * d - b_ * b_
        if abs(det) < 1e-30:
            return 0.0, 0.0
        inv = 1.0 / det
        return (inv * (d * rhs_x - b_ * rhs_y),
                inv * (-b_ * rhs_x + a * rhs_y))
    A = np.column_stack([dx, dy])
    x, *_ = np.linalg.lstsq(A, db, rcond=None)
    return float(x[0]), float(x[1])


def flux_interior(rho_P, u_P, v_P, gamma_P, rho_A, u_A, v_A, gamma_A, n_x, n_y, area):
    rho_m = 0.5 * (rho_P + rho_A)
    u_m = 0.5 * (u_P + u_A)
    v_m = 0.5 * (v_P + v_A)
    n_dot_rhou = rho_m * (n_x * u_m + n_y * v_m)
    F_rho = n_dot_rhou * area
    if n_dot_rhou >= 0.0:
        F_u = n_dot_rhou * u_P * area
        F_v = n_dot_rhou * v_P * area
    else:
        F_u = n_dot_rhou * u_A * area
        F_v = n_dot_rhou * v_A * area
    n_dot_u = n_x * u_m + n_y * v_m
    if n_dot_u >= 0.0:
        F_gamma = n_dot_u * gamma_P * area
    else:
        F_gamma = n_dot_u * gamma_A * area
    F_gamma1 = n_dot_u * area
    return F_rho, F_u, F_v, F_gamma, F_gamma1


def flux_boundary(rho_P, u_P, v_P, gamma_P, rho_A, u_A, v_A, gamma_A, n_x, n_y, area, n_dot_u):
    rho_m = 0.5 * (rho_P + rho_A)
    n_dot_rhou = rho_m * n_dot_u
    F_rho = n_dot_rhou * area
    if n_dot_u <= 0.0:
        F_u = n_dot_rhou * u_A * area
        F_v = n_dot_rhou * v_A * area
        F_gamma = n_dot_u * gamma_A * area
    else:
        F_u = n_dot_rhou * u_P * area
        F_v = n_dot_rhou * v_P * area
        F_gamma = n_dot_u * gamma_P * area
    F_gamma1 = n_dot_u * area
    return F_rho, F_u, F_v, F_gamma, F_gamma1


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Numba-jitted inner kernel (massive speedup vs pure Python)
# --------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        # no-op decorator if numba is not installed
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(f): return f
        return wrap


@njit(cache=True, fastmath=True)
def _step_jit(
    i_model,
    ct, nbrs, volume, cc_to_cc_x, cc_to_cc_y,
    face_nx, face_ny, face_area,
    T11, T12, T21, T22,
    Kxx_base, Kxy_base, Kyy_base, perm_factor, viscosity,
    phi_old, phi_new,
    rho, u, v, p, gamma_vof,
    dt,
    # i_model=1 EOS
    ap0, ap1, ap2,
    # i_model=2,3 EOS
    p_a_eos, p_init_eos, c_eos, exp_eos, rho_air, rho_resin,
):
    """
    One explicit step.

    Continuity is written in the unified form

        rho_n = rho * phi_old/phi_new - dt * F_rho / (V * phi_new)

    For i_model=1 phi_old = phi_new = 1 -> rho_n = rho - dt*F/V.
    For i_model=2 phi_old = phi_new = phi (constant) -> rho_n = rho - dt*F/(V*phi).
    For i_model=3 phi_old, phi_new are the relaxed effective porosity
                  before/after the step (preform compaction).

    Momentum uses an implicit 2x2 Darcy drag with K = K_base * perm_factor[i].
    EOS branches by i_model (quadratic table for 1; power-law for 2/3).
    VOF: continuous flux update for 1; hard cutoff for 2/3.
    """
    N = rho.size
    K = nbrs.shape[1]

    rho_new = rho.copy()
    u_new = u.copy()
    v_new = v.copy()
    p_new = p.copy()
    g_new = gamma_vof.copy()
    rho_thresh = 0.5 * rho_resin

    for i in range(N):
        cti = ct[i]
        if cti != 1 and cti != -3:
            continue

        # --- LSQ pressure gradient (2x2 normal equations) ---
        a = 0.0
        b_ = 0.0
        d = 0.0
        rx = 0.0
        ry = 0.0
        nb_count = 0
        for k in range(K):
            j = nbrs[i, k]
            if j < 0:
                break
            dx = cc_to_cc_x[i, k]
            dy = cc_to_cc_y[i, k]
            db = p[j] - p[i]
            a  += dx * dx
            b_ += dx * dy
            d  += dy * dy
            rx += dx * db
            ry += dy * db
            nb_count += 1

        if nb_count < 2:
            dpdx = 0.0
            dpdy = 0.0
        else:
            det = a * d - b_ * b_
            if det < 1e-30 and det > -1e-30:
                dpdx = 0.0
                dpdy = 0.0
            else:
                inv = 1.0 / det
                dpdx = inv * (d * rx - b_ * ry)
                dpdy = inv * (-b_ * rx + a * ry)

        F_rho = 0.0
        F_u = 0.0
        F_v = 0.0
        F_g = 0.0
        F_g1 = 0.0

        pf = perm_factor[i]
        kxx = Kxx_base[i] * pf
        kxy = Kxy_base[i] * pf
        kyy = Kyy_base[i] * pf
        mu = viscosity[i]

        for k in range(K):
            j = nbrs[i, k]
            if j < 0:
                break
            u_j = T11[i, k] * u[j] + T12[i, k] * v[j]
            v_j = T21[i, k] * u[j] + T22[i, k] * v[j]
            nx = face_nx[i, k]
            ny = face_ny[i, k]
            A = face_area[i, k]

            ctj = ct[j]
            if ctj == 1 or ctj == -3:
                rho_m = 0.5 * (rho[i] + rho[j])
                u_m   = 0.5 * (u[i] + u_j)
                v_m   = 0.5 * (v[i] + v_j)
                n_dot_rhou = rho_m * (nx * u_m + ny * v_m)
                fR = n_dot_rhou * A
                if n_dot_rhou >= 0.0:
                    fU = n_dot_rhou * u[i] * A
                    fV = n_dot_rhou * v[i] * A
                else:
                    fU = n_dot_rhou * u_j * A
                    fV = n_dot_rhou * v_j * A
                n_dot_u = nx * u_m + ny * v_m
                if n_dot_u >= 0.0:
                    fG = n_dot_u * gamma_vof[i] * A
                else:
                    fG = n_dot_u * gamma_vof[j] * A
                fG1 = n_dot_u * A
            else:
                if ctj == -2:
                    n_dot_u = nx * u[i] + ny * v[i]
                else:
                    qx = -(kxx * dpdx + kxy * dpdy) / mu
                    qy = -(kxy * dpdx + kyy * dpdy) / mu
                    val = nx * qx + ny * qy
                    n_dot_u = val if val < 0.0 else 0.0
                rho_m = 0.5 * (rho[i] + rho[j])
                n_dot_rhou = rho_m * n_dot_u
                fR = n_dot_rhou * A
                if n_dot_u <= 0.0:
                    fU = n_dot_rhou * u[j] * A
                    fV = n_dot_rhou * v[j] * A
                    fG = n_dot_u * gamma_vof[j] * A
                else:
                    fU = n_dot_rhou * u[i] * A
                    fV = n_dot_rhou * v[i] * A
                    fG = n_dot_u * gamma_vof[i] * A
                fG1 = n_dot_u * A

            F_rho += fR
            F_u += fU
            F_v += fV
            F_g += fG
            F_g1 += fG1

        vol = volume[i]

        # --- Continuity (unified form across models) ---
        pn = phi_new[i]
        po = phi_old[i]
        rn = (rho[i] * po - dt * F_rho / vol) / pn
        if rn < 0.0:
            rn = 0.0
        rho_new[i] = rn

        # --- Momentum: full 2x2 implicit drag ---
        det_K = kxx * kyy - kxy * kxy
        if det_K < 1e-40 and det_K > -1e-40:
            inv_kxx = 1e30
            inv_kxy = 0.0
            inv_kyy = 1e30
        else:
            inv_det = 1.0 / det_K
            inv_kxx =  kyy * inv_det
            inv_kxy = -kxy * inv_det
            inv_kyy =  kxx * inv_det

        Mxx = rn + dt * mu * inv_kxx
        Mxy = dt * mu * inv_kxy
        Myy = rn + dt * mu * inv_kyy

        rhs_u = rho[i] * u[i] - dt * F_u / vol - dt * dpdx
        rhs_v = rho[i] * v[i] - dt * F_v / vol - dt * dpdy

        det_M = Mxx * Myy - Mxy * Mxy
        if det_M < 1e-40 and det_M > -1e-40:
            u_new[i] = 0.0
            v_new[i] = 0.0
        else:
            inv_dM = 1.0 / det_M
            u_new[i] = ( Myy * rhs_u - Mxy * rhs_v) * inv_dM
            v_new[i] = (-Mxy * rhs_u + Mxx * rhs_v) * inv_dM

        # --- VOF and EOS ---
        if i_model == 1:
            g_raw = (pn * gamma_vof[i]
                     - dt * (F_g - gamma_vof[i] * F_g1) / vol) / pn
            if g_raw < 0.0:
                g_raw = 0.0
            if g_raw > 1.0:
                g_raw = 1.0
            g_new[i] = g_raw
            p_new[i] = ap0 * rn * rn + ap1 * rn + ap2
        else:
            # Hard VOF cutoff for two-fluid surrogate models
            g_new[i] = 1.0 if rn >= rho_thresh else 0.0
            # Power-law EOS, with d_rho clamped to (rho_resin - rho_air)
            # to keep the (potentially overshooting) transient rho from
            # blowing up float64 in d_rho**exp_eos. The result is
            # capped to p_a anyway, so clamping d_rho first is a
            # consistent, overflow-safe rewriting of the same EOS.
            d_rho = rn - rho_air
            drho_max = rho_resin - rho_air
            if d_rho < 0.0:
                pwr = 0.0
            elif d_rho > drho_max:
                pwr = drho_max ** exp_eos
            else:
                pwr = d_rho ** exp_eos
            pp = p_init_eos + c_eos * pwr
            if pp < p_init_eos:
                pp = p_init_eos
            if pp > p_a_eos:
                pp = p_a_eos
            p_new[i] = pp

    for i in range(N):
        cti = ct[i]
        if cti == -1:
            rho_new[i] = rho[i]
            p_new[i] = p[i]
            g_new[i] = 1.0
            u_new[i] = 0.0
            v_new[i] = 0.0
        elif cti == -2:
            rho_new[i] = rho[i]
            p_new[i] = p[i]
            g_new[i] = 0.0
            u_new[i] = 0.0
            v_new[i] = 0.0

    for i in range(N):
        rho[i] = rho_new[i]
        u[i] = u_new[i]
        v[i] = v_new[i]
        p[i] = p_new[i]
        gamma_vof[i] = g_new[i]


# --------------------------------------------------------------------------
# Thermal / cure helpers + kernel
# --------------------------------------------------------------------------
R_GAS = 8.314462618  # [J/(mol K)]


@njit(cache=True, fastmath=True)
def _viscosity_TA(T, alpha, mu_inf, E_mu, alpha_gel,
                  C1, C2, mu_max, cure_on):
    """
    Castro-Macosko viscosity. Vectorised in-place semantics: returns a new
    array. When cure_on=False, only the Arrhenius term is applied.

    The Arrhenius exponent is clamped to a value that keeps exp() finite
    (~700 in float64) so unreasonably cold T just returns mu_max instead
    of raising an overflow warning.
    """
    N = T.size
    out = np.empty(N)
    EXP_MAX = 700.0
    for i in range(N):
        Ti = T[i]
        if Ti < 1.0:
            Ti = 1.0
        arg = E_mu / (R_GAS * Ti)
        if arg > EXP_MAX:
            mu = mu_max
        else:
            mu = mu_inf * np.exp(arg)
        if cure_on:
            ai = alpha[i]
            if ai >= alpha_gel:
                mu = mu_max
            else:
                ratio = alpha_gel / (alpha_gel - ai)
                expn = C1 + C2 * ai
                mu = mu * ratio ** expn
        if mu > mu_max:
            mu = mu_max
        if mu < 1e-6:
            mu = 1e-6
        out[i] = mu
    return out


@njit(cache=True, fastmath=True)
def _cure_rate(T, alpha, A1, A2, E1, E2, m, n):
    """Kamal-Sourour: dα/dt = (k1 + k2 α^m)(1-α)^n. Returns array."""
    N = T.size
    out = np.empty(N)
    for i in range(N):
        Ti = T[i]
        if Ti < 1.0:
            Ti = 1.0
        ai = alpha[i]
        if ai < 0.0:
            ai = 0.0
        if ai > 1.0:
            ai = 1.0
        k1 = A1 * np.exp(-E1 / (R_GAS * Ti))
        k2 = A2 * np.exp(-E2 / (R_GAS * Ti))
        out[i] = (k1 + k2 * ai ** m) * (1.0 - ai) ** n
    return out


@njit(cache=True, fastmath=True)
def _step_thermal_jit(
    ct, nbrs, volume,
    face_nx, face_ny, face_area,
    T11, T12, T21, T22,
    thickness, porosity, gamma_vof,
    rho_resin, cp_resin, rho_fiber, cp_fiber,
    h_tool, T_tool,
    cure_on, H_total,
    A1, A2, E1, E2, m_kin, n_kin,
    u, v, T, alpha,
    dt,
):
    """
    One explicit step of the lumped-1D thermal model + cure update.

    Per cell:
      (rho cp)_eff dT/dt + (rho cp)_resin * phi * gamma * (u . grad T)
            = -2 h_tool / h_thk * (T - T_tool)            (1D sink)
              + gamma * phi * rho_resin * H_total * dα/dt (cure source)

    Tool sink is integrated implicitly per cell to allow large dt.
    Convection is upwind on the resin-bearing fraction (gamma weighted).
    In-plane conduction is omitted in this first attempt (negligible for
    thin shells; can be added by symmetry with the pressure-gradient
    block in _step_jit).
    """
    N = T.size
    K = nbrs.shape[1]
    T_new = T.copy()
    a_new = alpha.copy()

    for i in range(N):
        cti = ct[i]
        if cti != 1 and cti != -3:
            continue

        h_thk = thickness[i]
        phi = porosity[i]
        g_i = gamma_vof[i]

        # Effective volumetric heat capacity. Resin contribution is
        # weighted by gamma so dry preform sees only fiber thermal mass.
        rcp_resin = phi * g_i * rho_resin * cp_resin
        rcp_fiber = (1.0 - phi) * rho_fiber * cp_fiber
        rcp = rcp_resin + rcp_fiber
        if rcp < 1e-12:
            continue

        # --- Convective term in NON-CONSERVATIVE form ---
        # rcp_resin * g * (u . grad T) is discretised as a sum over faces of
        # rho*cp * g_face * ndu * (T_face - T_i) * A / vol. This vanishes
        # identically when T is uniform, which is the correct behaviour
        # for compressible LCM filling where (rho cp)_eff does not track
        # the changing resin mass during the step.
        F_T = 0.0
        for k in range(K):
            j = nbrs[i, k]
            if j < 0:
                break
            u_j = T11[i, k] * u[j] + T12[i, k] * v[j]
            v_j = T21[i, k] * u[j] + T22[i, k] * v[j]
            nx = face_nx[i, k]
            ny = face_ny[i, k]
            A = face_area[i, k]
            u_m = 0.5 * (u[i] + u_j)
            v_m = 0.5 * (v[i] + v_j)
            ndu = nx * u_m + ny * v_m
            ctj = ct[j]
            if ctj == -2:
                # outflow: zero-gradient on T → no contribution
                continue
            if ctj == -1:
                # inlet face: only inflow contributes (resin enters carrying T_inlet)
                if ndu < 0.0:
                    F_T += rho_resin * cp_resin * 1.0 * ndu * (T[j] - T[i]) * A
                continue
            # interior / wall neighbour: upwind T_face, gamma_face
            if ndu >= 0.0:
                g_face = g_i
                T_face = T[i]
            else:
                g_face = gamma_vof[j]
                T_face = T[j]
            F_T += rho_resin * cp_resin * g_face * ndu * (T_face - T[i]) * A

        vol = volume[i]

        # --- Cure source term ---
        if cure_on:
            ai = a_new[i]
            if ai < 0.0:
                ai = 0.0
            if ai > 1.0:
                ai = 1.0
            Ti = T[i]
            if Ti < 1.0:
                Ti = 1.0
            k1 = A1 * np.exp(-E1 / (R_GAS * Ti))
            k2 = A2 * np.exp(-E2 / (R_GAS * Ti))
            da_dt = (k1 + k2 * ai ** m_kin) * (1.0 - ai) ** n_kin
            # Cap step to avoid runaway
            da = da_dt * dt
            if da > 0.05:
                da = 0.05
            a_new[i] = ai + da
            if a_new[i] > 1.0:
                a_new[i] = 1.0
            S_cure = g_i * phi * rho_resin * H_total * (da / dt)
        else:
            S_cure = 0.0

        # --- Tool sink + convection + source, implicit on the sink ---
        # rcp * (T_new - T) / dt = -F_T/vol - 2 h/h_thk*(T_new - T_tool) + S
        # => T_new (rcp/dt + 2h/h_thk) = rcp/dt * T - F_T/vol
        #                              + 2h/h_thk * T_tool + S
        sink = 2.0 * h_tool / max(h_thk, 1e-9)
        lhs = rcp / dt + sink
        rhs = (rcp / dt) * T[i] - F_T / vol + sink * T_tool + S_cure
        T_new[i] = rhs / lhs

    # BC: pin inlet cells; outlet/wall handled by skip + flux logic
    for i in range(N):
        if ct[i] == -1:
            T_new[i] = T[i]      # inlet T already set externally
            a_new[i] = alpha[i]  # inlet alpha pinned

    for i in range(N):
        T[i] = T_new[i]
        alpha[i] = a_new[i]


def _setup_eos_model1(params, p_a, p_init):
    """Compressible-air EOS lookup-table coefficients and inlet/init densities."""
    kappa = params.p_ref / (params.rho_ref ** params.gamma)
    p_int = np.array([0.0, 0.5e5, 1.0e5])
    rho_int = (p_int / kappa) ** (1.0 / params.gamma)
    A = np.column_stack([rho_int ** 2, rho_int, np.ones(3)])
    ap = np.linalg.solve(A, p_int)
    rho_a = (p_a / kappa) ** (1.0 / params.gamma)
    rho_init = (p_init / kappa) ** (1.0 / params.gamma)
    return ap, rho_a, rho_init


def _setup_eos_model23(params, exp_eos):
    """Two-fluid surrogate p(rho) = p_init + c*(rho-rho_air)^exp_eos.

    p_a, p_init are the absolute pressures (not normalized).
    Returns (c_eos, rho_a, rho_init).
    """
    p_a = params.p_inlet
    p_init = params.p_init
    drho = params.rho_resin - params.rho_air
    c = (p_a - p_init) / (drho ** exp_eos)
    return c, params.rho_resin, params.rho_air


def _initial_conditions(N, celltype, rho_a, rho_init, p_a, p_init):
    rho = np.full(N, rho_init)
    u = np.zeros(N)
    v = np.zeros(N)
    p = np.full(N, p_init)
    gamma_vof = np.zeros(N)
    inlet = celltype == CELL_INLET
    outlet = celltype == CELL_OUTLET
    rho[inlet] = rho_a
    p[inlet] = p_a
    gamma_vof[inlet] = 1.0
    rho[outlet] = rho_init
    p[outlet] = p_init
    gamma_vof[outlet] = 0.0
    return rho, u, v, p, gamma_vof


def _eigmax_K(Kxx, Kxy, Kyy):
    diff = Kxx - Kyy
    return 0.5 * (Kxx + Kyy + np.sqrt(diff * diff + 4.0 * Kxy * Kxy))


def run_filling(mesh, params, on_snapshot=None):
    """
    Main driver. Returns list of Snapshot objects; calls
    on_snapshot(snap) after each snapshot if provided.

    Three model branches share the same JIT kernel; per-cell arrays
    are precomputed here so the kernel sees a uniform interface:

      * Kxx_base, Kxy_base, Kyy_base : per-element K from build_stack_tensor
      * perm_factor[i]               : 1.0 for models 1/2; phi^3/(1-phi)^2 for 3
      * phi_old[i], phi_new[i]       : 1.0 (model 1); phi (model 2);
                                       relaxed effective porosity (model 3)
      * EOS scalars                  : ap0/ap1/ap2 (model 1) and
                                       p_a/p_init/c/exp/rho_air/rho_resin (2,3)
    """
    params.validate()
    neighbours, celltype = create_faces(mesh, params.max_neighbours)
    celltype, thickness, porosity, viscosity = \
        assign_parameters(mesh, params, celltype)

    geom = create_coordinate_systems(
        mesh, neighbours, celltype, thickness, params.max_neighbours,
    )

    Kxx_base, Kxy_base, Kyy_base, phi0, c_porosity = \
        build_stack_tensor(mesh, params.stack)

    N = mesh.N
    if not (celltype == CELL_INLET).any():
        raise RuntimeError("No inlet cells defined")

    # ---- choose EOS exponent (auto-bump for race-tracking) ----
    K_eig = _eigmax_K(Kxx_base, Kxy_base, Kyy_base)
    K_max = float(np.max(K_eig))
    K_min = float(np.min(K_eig[K_eig > 0])) if np.any(K_eig > 0) else K_max
    perm_ratio = K_max / max(K_min, 1e-30)
    if params.exp_eos > 0:
        exp_eos = int(params.exp_eos)
    else:
        exp_eos = 25 if perm_ratio >= 100.0 else 4
    betat2_fac = 0.1 if perm_ratio >= 100.0 else 1.0

    # ---- pressure normalization & EOS coefficients ----
    if params.i_model == 1:
        p_eps = 100.0
        p_a = params.p_inlet - params.p_init + p_eps
        p_init_run = p_eps
        ap, rho_a, rho_init = _setup_eos_model1(params, p_a, p_init_run)
        ap0, ap1, ap2 = float(ap[0]), float(ap[1]), float(ap[2])
        c_eos = 0.0
        p_a_eos = p_a
        p_init_eos = p_init_run
        rho_air_param = params.rho_air
        rho_resin_param = params.rho_resin
    else:
        # i_model 2 or 3: use absolute pressures, two-fluid EOS
        p_a = params.p_inlet
        p_init_run = params.p_init
        c_eos, rho_a, rho_init = _setup_eos_model23(params, exp_eos)
        ap0 = ap1 = ap2 = 0.0
        p_a_eos = p_a
        p_init_eos = p_init_run
        rho_air_param = params.rho_air
        rho_resin_param = params.rho_resin

    rho, u, v, p, gamma_vof = _initial_conditions(
        N, celltype, rho_a, rho_init, p_a, p_init_run)

    # ---- per-cell porosity / perm-factor scratch arrays ----
    if params.i_model == 1:
        phi_old = np.ones(N)
        phi_new = np.ones(N)
        perm_factor = np.ones(N)
    elif params.i_model == 2:
        phi_old = porosity.copy()
        phi_new = porosity.copy()
        perm_factor = np.ones(N)
    else:  # i_model == 3
        # Initialize phi_eff_old to the volume-conserving target evaluated
        # at the initial pressure. For all interior cells p == p_init,
        # so phi_p == phi0 + c*p_init^2 (small perturbation).
        phi_p_init = np.clip(phi0 + c_porosity * p_init_run ** 2, 1e-6, 0.999)
        phi_target_init = (1.0 - phi0) / (1.0 - phi_p_init) * phi_p_init
        phi_old = phi_target_init.copy()
        phi_new = phi_target_init.copy()
        perm_factor = phi_p_init ** 3 / (1.0 - phi_p_init) ** 2

    # ---- thermal / cure state ----
    thermal_on = bool(params.thermal_enabled)
    cure_on = bool(params.cure_enabled)
    if thermal_on:
        T_field = np.full(N, params.T_init, dtype=np.float64)
        T_field[celltype == CELL_INLET] = params.T_inlet
        alpha_field = np.full(N, params.alpha_init, dtype=np.float64)
        # Refresh viscosity from the initial (T, alpha) so the CFL below
        # uses a consistent mu_min instead of params.mu_resin.
        viscosity = _viscosity_TA(
            T_field, alpha_field,
            params.mu_inf, params.E_mu, params.alpha_gel,
            params.C1_visc, params.C2_visc, params.mu_max, cure_on,
        )
    else:
        T_field = np.empty(0)
        alpha_field = np.empty(0)

    # ---- initial dt (CFL on Darcy-driven max velocity) ----
    area_min = float(np.min(geom.volume / thickness))
    h_min = np.sqrt(area_min)
    dp = p_a - p_init_run
    K_eff_max = K_max  # eigmax already reflects in-plane stack tensor
    if params.i_model == 3:
        K_eff_max *= float(np.max(perm_factor))
    mu_min = float(np.min(viscosity))
    u_max = K_eff_max * dp / (mu_min * h_min) + 1e-30
    dt = params.cfl * betat2_fac * h_min / u_max
    # Thermal stability cap: dt <= cp_eff * h_thk / (2 * h_tool) keeps the
    # implicit Newton-cooling step well-conditioned even though it's
    # unconditionally stable. Convective CFL on T is already covered by
    # the flow CFL since T is advected with u.
    if thermal_on and params.h_tool > 0:
        h_thk_min = float(np.min(thickness))
        rcp_min = (params.rho_resin * params.cp_resin
                   if params.rho_resin > 0 else 1.0e6)
        dt_therm = 0.5 * rcp_min * h_thk_min / (2.0 * params.h_tool + 1e-30)
        dt = min(dt, dt_therm)

    n_pics = max(4, (params.n_pics // 4) * 4)
    t_max = max(params.tmax, n_pics * dt)
    dt_snap = t_max / n_pics
    # Absolute dt cap: at least 40 steps per snapshot interval. Two-
    # fluid models (i_model 2/3) have stiffer dynamics during transient
    # ramp-up, so the cap is the only thing keeping rho from
    # overshooting rho_resin in the first few steps.
    dt_pic_max = t_max / max(40 * n_pics, 1)
    dt = min(dt, dt_pic_max)
    dt0 = dt
    # Growth multiplier for the adaptive dt: model 1 is forgiving, the
    # two-fluid models are not.
    dt_growth = 1000.0 if params.i_model == 1 else 50.0

    # Absolute-pressure offset: i_model=1 stores p in normalized form
    # (origin shifted by p_eps), i_model=2/3 store absolute pressure.
    if params.i_model == 1:
        p_offset = float(params.p_init - p_init_run)
    else:
        p_offset = 0.0

    snapshots = []
    def take_snapshot(step, t):
        if params.i_model == 3:
            phi_p_now = np.clip(phi0 + c_porosity * p ** 2, 1e-3, 0.95)
            t_compact = (1.0 - phi0) / (1.0 - phi_p_now) * thickness
            snap = Snapshot(step=step, t=t, gamma=gamma_vof.copy(),
                            p=p.copy(), celltype=celltype.copy(),
                            porosity=phi_p_now, thickness=t_compact,
                            p_offset=p_offset)
        else:
            snap = Snapshot(step=step, t=t, gamma=gamma_vof.copy(),
                            p=p.copy(), celltype=celltype.copy(),
                            p_offset=p_offset)
        if thermal_on:
            snap.T = T_field.copy()
            snap.alpha = alpha_field.copy()
            snap.mu = viscosity.copy()
        snapshots.append(snap)
        if on_snapshot is not None:
            on_snapshot(snap)

    # Pending cascade events, sorted by activation time. At each step we
    # flip the listed cells to CELL_INLET and pin their state to inlet
    # values; the kernel then leaves them alone for the rest of the run.
    pending_cascade = sorted(
        [(float(t_a), np.asarray(cids, dtype=int))
         for t_a, cids in (params.cascade_events or [])],
        key=lambda e: e[0],
    )

    def _activate_cascade(cids):
        celltype[cids] = CELL_INLET
        rho[cids] = rho_a
        p[cids] = p_a
        gamma_vof[cids] = 1.0
        u[cids] = 0.0
        v[cids] = 0.0
        if thermal_on:
            T_field[cids] = params.T_inlet
            alpha_field[cids] = params.alpha_init

    take_snapshot(0, 0.0)
    t_next = dt_snap
    t = 0.0
    step = 0
    while t <= t_max:
        # Activate any cascade injection ports whose t_activate has passed.
        while pending_cascade and t >= pending_cascade[0][0]:
            _, cids = pending_cascade.pop(0)
            _activate_cascade(cids)

        # i_model=3: refresh porosity / permeability factor before the step
        if params.i_model == 3:
            phi_p = np.clip(phi0 + c_porosity * p ** 2, 1e-3, 0.95)
            np.power(phi_p, 3, out=perm_factor)
            perm_factor /= (1.0 - phi_p) ** 2
            phi_target = (1.0 - phi0) / (1.0 - phi_p) * phi_p
            # 1% relaxation, matches Julia LCMsim
            phi_new[:] = phi_old + 0.01 * (phi_target - phi_old)

        # Refresh viscosity from current (T, alpha) before the flow step
        # so Darcy drag sees the temperature/cure-shifted mu.
        if thermal_on:
            viscosity = _viscosity_TA(
                T_field, alpha_field,
                params.mu_inf, params.E_mu, params.alpha_gel,
                params.C1_visc, params.C2_visc, params.mu_max, cure_on,
            )

        _step_jit(
            params.i_model,
            geom.celltype, geom.neighbours, geom.volume,
            geom.cc_to_cc_x, geom.cc_to_cc_y,
            geom.face_nx, geom.face_ny, geom.face_area,
            geom.T11, geom.T12, geom.T21, geom.T22,
            Kxx_base, Kxy_base, Kyy_base, perm_factor, viscosity,
            phi_old, phi_new,
            rho, u, v, p, gamma_vof,
            dt,
            ap0, ap1, ap2,
            p_a_eos, p_init_eos, c_eos, exp_eos, rho_air_param, rho_resin_param,
        )
        if params.i_model == 3:
            phi_old[:] = phi_new

        if thermal_on:
            _step_thermal_jit(
                geom.celltype, geom.neighbours, geom.volume,
                geom.face_nx, geom.face_ny, geom.face_area,
                geom.T11, geom.T12, geom.T21, geom.T22,
                thickness, porosity, gamma_vof,
                params.rho_resin, params.cp_resin,
                params.rho_fiber, params.cp_fiber,
                params.h_tool, params.T_tool,
                cure_on, params.H_total,
                params.A1_cure, params.A2_cure,
                params.E1_cure, params.E2_cure,
                params.m_cure, params.n_cure,
                u, v, T_field, alpha_field,
                dt,
            )

        step += 1
        t += dt
        if step > 4:
            interior = (celltype == CELL_INTERIOR) | (celltype == CELL_WALL)
            speed = np.sqrt(u[interior]**2 + v[interior]**2) + 1e-12
            h = np.sqrt(geom.volume[interior] / thickness[interior])
            dt_conv = params.cfl * betat2_fac * float(np.min(h / speed))
            dt = min(max(dt_conv, dt0), dt_growth * dt0, dt_pic_max)
        if t >= t_next or t + dt > t_max:
            take_snapshot(step, t)
            t_next += dt_snap
        interior = (celltype == CELL_INTERIOR) | (celltype == CELL_WALL)
        if gamma_vof[interior].mean() > 0.985:
            take_snapshot(step, t)
            break
    return snapshots
