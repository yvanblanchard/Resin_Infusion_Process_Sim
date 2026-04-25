"""
RTMsim-Py: Python port of RTMsim (Obertscheider et al., FHWN).

Single-file implementation of the Finite Area Method solver for RTM
filling simulation on triangle shell meshes. Physics: compressible
Euler + Darcy drag + VOF. Numerics: least-squares gradient + first-
order upwind flux, explicit Euler for continuity/VOF with implicit
Darcy drag in the momentum update.

This port drops GUI, Nastran BDF reader, JLD2 file I/O, and post-
processing plots from the original package. Meshes are built from
numpy arrays; up to 4 in-plane patches define inlets, outlets, and
preform regions with distinct orientations.
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
    thickness: float = 3e-3
    porosity: float = 0.7
    K1: float = 3e-10
    alpha: float = 1.0
    refdir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))


@dataclass
class PlyProperties:
    """
    One ply in a stacked-laminate (PCOMP) element.

    Each ply has its own thickness, in-plane principal permeabilities
    (K1 along the fibre, K2 across), and a global-frame fibre direction.
    Porosity is per-ply because different fabrics in the stack can have
    different fibre volume fractions.
    """
    thickness: float = 0.75e-3
    porosity: float = 0.7
    K1: float = 3e-10
    K2: float = 0.6e-10
    refdir: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0]))


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

    This is the standard LCM PCOMP shell assumption: same in-plane
    pressure gradient across all plies (parallel flow), so flow rate
    adds — equivalent to thickness-weighted tensor averaging.
    """
    plies: list = field(default_factory=list)

    @property
    def total_thickness(self) -> float:
        return float(sum(p.thickness for p in self.plies))

    @property
    def effective_porosity(self) -> float:
        """Volume-weighted porosity: sum(phi_p * t_p) / sum(t_p)."""
        t = self.total_thickness
        if t == 0:
            return 0.0
        return float(sum(p.porosity * p.thickness for p in self.plies) / t)


@dataclass
class SimParameters:
    i_model: int = 1
    tmax: float = 200.0
    p_ref: float = 1.01325e5
    rho_ref: float = 1.225
    gamma: float = 1.4
    mu_resin: float = 0.06
    p_inlet: float = 1.35e5
    p_init: float = 1.0e5
    main: PatchProperties = field(default_factory=PatchProperties)
    patches: List[PatchProperties] = field(
        default_factory=lambda: [PatchProperties() for _ in range(4)])
    patch_types: List[int] = field(
        default_factory=lambda: [PATCH_INLET, PATCH_IGNORE, PATCH_IGNORE, PATCH_IGNORE])
    n_pics: int = 16
    max_neighbours: int = 10
    gradient_method: int = 3
    cfl: float = 0.05
    # If provided, this stack overrides PatchProperties for permeability:
    # every element gets the same stack, and per-element permeability is
    # the thickness-weighted sum of rotated ply tensors.
    stack: object = None

    def validate(self):
        if self.i_model != 1:
            raise ValueError("Only i_model=1 is implemented")
        if self.tmax <= 0:
            raise ValueError("tmax must be > 0")
        if self.p_inlet <= self.p_init:
            raise ValueError("p_inlet must be > p_init")


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
    Overlay patch types / scalar properties on each cell.

    Note: when params.stack is provided, the per-element permeability
    and refdir returned here are placeholders only. The real per-element
    tensor is computed by build_stack_tensor() later.
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

    if params.stack is not None:
        # Stacked-laminate mode: one stack covers every element. Use
        # total stack thickness and effective porosity for all cells.
        t_tot = params.stack.total_thickness
        phi_eff = params.stack.effective_porosity
        thickness = np.full(N, t_tot)
        porosity  = np.full(N, phi_eff)
        # `perm` and `alpha` are unused in stack mode (the solver reads
        # K_eff arrays instead). Keep harmless placeholders.
        perm  = np.full(N, params.stack.plies[0].K1 if params.stack.plies else 1e-10)
        alpha = np.ones(N)
        # refdir is also unused in stack mode but the coord-system
        # builder still reads it. Use +x; with theta_used=0 below,
        # this has no effect on the local frame.
        refdir = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
        viscosity = np.full(N, params.mu_resin)
        return ct, thickness, porosity, perm, alpha, refdir, viscosity

    # Legacy patch-based mode
    thickness = np.full(N, params.main.thickness)
    porosity  = np.full(N, params.main.porosity)
    perm      = np.full(N, params.main.K1)
    alpha     = np.full(N, params.main.alpha)
    refdir    = np.tile(_unit(params.main.refdir), (N, 1))
    viscosity = np.full(N, params.mu_resin)
    for pi in range(4):
        if params.patch_types[pi] != PATCH_PREFORM:
            continue
        cells = mesh.patch_cell_ids[pi]
        if cells.size == 0:
            continue
        pp = params.patches[pi]
        thickness[cells] = pp.thickness
        porosity[cells]  = pp.porosity
        perm[cells]      = pp.K1
        alpha[cells]     = pp.alpha
        refdir[cells]    = _unit(pp.refdir)
    return ct, thickness, porosity, perm, alpha, refdir, viscosity


def create_coordinate_systems(mesh, neighbours, celltype, thickness, refdir, max_neighbours, use_theta=True):
    """Per-cell basis aligned with projected refdir; flattened neighbour geometry."""
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
        Tm = np.column_stack([e1, e2, e3])
        if use_theta:
            r_loc = np.linalg.solve(Tm, refdir[ind])
            th = np.arctan2(r_loc[1], r_loc[0])
        else:
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

def build_stack_tensor(mesh, stack, refdir_per_ply_used):
    """
    Per-element 2x2 in-plane permeability tensor for a stacked laminate.

    For each element, build the local geometric frame (e1, e2, e3)
    from the triangle nodes (same as in create_coordinate_systems with
    theta=0). For each ply, project its global refdir into that
    element's tangent plane to get the angle alpha_p in (e1, e2). Then
    the ply's tensor in the element frame is

        R(alpha_p) . diag(K1_p, K2_p) . R(alpha_p)^T

    The element-effective tensor is the thickness-weighted sum of all
    ply tensors, divided by total stack thickness. Returns Kxx, Kxy,
    Kyy arrays (length N).
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
            # project refdir into element tangent plane
            rx = float(r @ e1)
            ry = float(r @ e2)
            mag = (rx * rx + ry * ry) ** 0.5
            if mag < 1e-30:
                # fibre is normal to element — no in-plane contribution;
                # default to e1 to avoid NaN
                rx = 1.0
                ry = 0.0
            else:
                rx /= mag
                ry /= mag
            # rotation from fibre frame to element frame:
            # K_elem = R . diag(K1, K2) . R^T  with R = [[c, -s], [s, c]]
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

    return Kxx, Kxy, Kyy




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
    ct, nbrs, volume, cc_to_cc_x, cc_to_cc_y,
    face_nx, face_ny, face_area,
    T11, T12, T21, T22,
    porosity, Kxx, Kxy, Kyy, viscosity,
    rho, u, v, p, gamma_vof,
    dt, ap0, ap1, ap2, gradient_method,
):
    """
    One explicit step with full 2x2 in-plane permeability tensor.

    Per-cell K is given by (Kxx, Kxy, Kyy) in the element local frame.
    The Darcy drag in the momentum update is treated implicitly as a
    2x2 system

        (rho_n I + dt * mu * K^{-1}) u_n = rho u - dt * F/V - dt * grad p

    The inlet inflow uses the full tensor:  q = -(K . grad p) / mu.
    """
    N = rho.size
    K = nbrs.shape[1]

    rho_new = rho.copy()
    u_new = u.copy()
    v_new = v.copy()
    p_new = p.copy()
    g_new = gamma_vof.copy()

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

        kxx = Kxx[i]
        kxy = Kxy[i]
        kyy = Kyy[i]
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

        # --- Continuity ---
        rn = rho[i] - dt * F_rho / vol
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

        phi = porosity[i]
        g_raw = (phi * gamma_vof[i]
                 - dt * (F_g - gamma_vof[i] * F_g1) / vol) / phi
        if g_raw < 0.0:
            g_raw = 0.0
        if g_raw > 1.0:
            g_raw = 1.0
        g_new[i] = g_raw

        p_new[i] = ap0 * rn * rn + ap1 * rn + ap2

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


def _setup_eos(params, p_a, p_init):
    kappa = params.p_ref / (params.rho_ref ** params.gamma)
    p_int = np.array([0.0, 0.5e5, 1.0e5])
    rho_int = (p_int / kappa) ** (1.0 / params.gamma)
    A = np.column_stack([rho_int ** 2, rho_int, np.ones(3)])
    ap = np.linalg.solve(A, p_int)
    rho_a = (p_a / kappa) ** (1.0 / params.gamma)
    rho_init = (p_init / kappa) ** (1.0 / params.gamma)
    return kappa, ap, rho_a, rho_init


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


def _step(geom, porosity, perm, alpha, viscosity,
          rho, u, v, p, gamma_vof, dt, ap, gradient_method):
    N = rho.size
    ct = geom.celltype
    rho_new = rho.copy()
    u_new = u.copy()
    v_new = v.copy()
    p_new = p.copy()
    gamma_new = gamma_vof.copy()
    nbrs = geom.neighbours
    cc_to_cc_x = geom.cc_to_cc_x
    cc_to_cc_y = geom.cc_to_cc_y
    face_nx = geom.face_nx
    face_ny = geom.face_ny
    face_area = geom.face_area
    T11 = geom.T11
    T12 = geom.T12
    T21 = geom.T21
    T22 = geom.T22
    for i in range(N):
        if ct[i] != CELL_INTERIOR and ct[i] != CELL_WALL:
            continue
        dpdx, dpdy = numerical_gradient(gradient_method, i, p, nbrs, cc_to_cc_x, cc_to_cc_y)
        F_rho = 0.0
        F_u = 0.0
        F_v = 0.0
        F_g = 0.0
        F_g1 = 0.0
        row_nb = nbrs[i]
        for k in range(row_nb.size):
            j = row_nb[k]
            if j < 0:
                break
            u_j = T11[i, k] * u[j] + T12[i, k] * v[j]
            v_j = T21[i, k] * u[j] + T22[i, k] * v[j]
            nx = face_nx[i, k]
            ny = face_ny[i, k]
            A = face_area[i, k]
            if ct[j] == CELL_INTERIOR or ct[j] == CELL_WALL:
                fR, fU, fV, fG, fG1 = flux_interior(
                    rho[i], u[i], v[i], gamma_vof[i],
                    rho[j], u_j, v_j, gamma_vof[j], nx, ny, A)
            else:
                if ct[j] == CELL_OUTLET:
                    n_dot_u = nx * u[i] + ny * v[i]
                else:
                    K1 = perm[i]
                    K2 = alpha[i] * perm[i]
                    mu = viscosity[i]
                    qx = -K1 / mu * dpdx
                    qy = -K2 / mu * dpdy
                    n_dot_u = min(0.0, nx * qx + ny * qy)
                fR, fU, fV, fG, fG1 = flux_boundary(
                    rho[i], u[i], v[i], gamma_vof[i],
                    rho[j], u[j], v[j], gamma_vof[j], nx, ny, A, n_dot_u)
            F_rho += fR
            F_u += fU
            F_v += fV
            F_g += fG
            F_g1 += fG1
        vol = geom.volume[i]
        rho_new[i] = max(0.0, rho[i] - dt * F_rho / vol)
        mu = viscosity[i]
        K1 = perm[i]
        K2 = alpha[i] * K1
        denom_u = rho_new[i] + mu / K1 * dt
        denom_v = rho_new[i] + mu / K2 * dt
        u_new[i] = (rho[i] * u[i] - dt * F_u / vol - dt * dpdx) / denom_u
        v_new[i] = (rho[i] * v[i] - dt * F_v / vol - dt * dpdy) / denom_v
        phi = porosity[i]
        gamma_new[i] = (phi * gamma_vof[i] - dt * (F_g - gamma_vof[i] * F_g1) / vol) / phi
        gamma_new[i] = min(1.0, max(0.0, gamma_new[i]))
        p_new[i] = ap[0] * rho_new[i] ** 2 + ap[1] * rho_new[i] + ap[2]
    inlet = ct == CELL_INLET
    outlet = ct == CELL_OUTLET
    rho_new[inlet] = rho[inlet]
    p_new[inlet] = p[inlet]
    gamma_new[inlet] = 1.0
    u_new[inlet] = 0.0
    v_new[inlet] = 0.0
    rho_new[outlet] = rho[outlet]
    p_new[outlet] = p[outlet]
    gamma_new[outlet] = 0.0
    u_new[outlet] = 0.0
    v_new[outlet] = 0.0
    rho[:] = rho_new
    u[:] = u_new
    v[:] = v_new
    p[:] = p_new
    gamma_vof[:] = gamma_new


def run_filling(mesh, params, on_snapshot=None):
    """
    Main driver. Returns list of Snapshot objects; calls
    on_snapshot(snap) after each snapshot if provided.

    Two operating modes:

    * Patch mode (default): per-element diagonal K = diag(K1, alpha*K1)
      aligned with the patch refdir. Each element gets one fibre angle
      from the patch it belongs to.

    * Stack mode (params.stack is a LaminateStack): the laminate is
      stacked through the thickness. Every element carries every ply,
      and per-element K_eff is the thickness-weighted sum of rotated
      ply tensors

          K_eff(i) = (1/t_tot) sum_p R(theta_p,i) diag(K1_p, K2_p) R^T t_p

      expressed in the element's geometric local frame. Only the per-
      ply refdir varies between plies; thickness, K1, K2, porosity may
      also vary per ply.
    """
    params.validate()
    neighbours, celltype = create_faces(mesh, params.max_neighbours)
    celltype, thickness, porosity, perm, alpha, refdir, viscosity = \
        assign_parameters(mesh, params, celltype)

    # In stack mode the per-element local frame is purely geometric.
    use_theta = (params.stack is None)
    geom = create_coordinate_systems(
        mesh, neighbours, celltype, thickness, refdir, params.max_neighbours,
        use_theta=use_theta,
    )

    # Build the per-element K tensor consumed by _step_jit.
    if params.stack is not None:
        Kxx, Kxy, Kyy = build_stack_tensor(mesh, params.stack, refdir)
    else:
        # Diagonal in element local frame: K = diag(K1, alpha*K1).
        Kxx = perm.copy()
        Kyy = perm * alpha
        Kxy = np.zeros_like(perm)

    N = mesh.N
    if not (celltype == CELL_INLET).any():
        raise RuntimeError("No inlet cells defined")

    # Pressure normalization for EOS table stability.
    p_eps = 100.0
    p_a = params.p_inlet - params.p_init + p_eps
    p_init = p_eps
    kappa, ap, rho_a, rho_init = _setup_eos(params, p_a, p_init)
    rho, u, v, p, gamma_vof = _initial_conditions(
        N, celltype, rho_a, rho_init, p_a, p_init)

    # Initial dt: use the largest eigenvalue of K_eff as effective
    # permeability. For a 2x2 symmetric tensor lambda1 = (kxx+kyy +
    # sqrt((kxx-kyy)^2 + 4 kxy^2)) / 2.
    area_min = float(np.min(geom.volume / thickness))
    h_min = np.sqrt(area_min)
    dp = params.p_inlet - params.p_init
    diff = Kxx - Kyy
    lam1 = 0.5 * (Kxx + Kyy + np.sqrt(diff * diff + 4.0 * Kxy * Kxy))
    K_eff = float(np.max(lam1))
    mu_min = float(np.min(viscosity))
    u_max = K_eff * dp / (mu_min * h_min) + 1e-30
    dt = params.cfl * h_min / u_max
    dt0 = dt

    n_pics = max(4, (params.n_pics // 4) * 4)
    t_max = max(params.tmax, n_pics * dt)
    dt_snap = t_max / n_pics

    snapshots = []
    def take_snapshot(step, t):
        snap = Snapshot(step=step, t=t, gamma=gamma_vof.copy(),
                        p=p.copy(), celltype=celltype.copy())
        snapshots.append(snap)
        if on_snapshot is not None:
            on_snapshot(snap)

    take_snapshot(0, 0.0)
    t_next = dt_snap
    t = 0.0
    step = 0
    while t <= t_max:
        _step_jit(
            geom.celltype, geom.neighbours, geom.volume,
            geom.cc_to_cc_x, geom.cc_to_cc_y,
            geom.face_nx, geom.face_ny, geom.face_area,
            geom.T11, geom.T12, geom.T21, geom.T22,
            porosity, Kxx, Kxy, Kyy, viscosity,
            rho, u, v, p, gamma_vof,
            dt, float(ap[0]), float(ap[1]), float(ap[2]),
            params.gradient_method,
        )
        step += 1
        t += dt
        if step > 4:
            interior = (celltype == CELL_INTERIOR) | (celltype == CELL_WALL)
            speed = np.sqrt(u[interior]**2 + v[interior]**2) + 1e-12
            h = np.sqrt(geom.volume[interior] / thickness[interior])
            dt_conv = params.cfl * float(np.min(h / speed))
            dt = min(max(dt_conv, dt0), 1000.0 * dt0)
        if t >= t_next or t + dt > t_max:
            take_snapshot(step, t)
            t_next += dt_snap
        interior = (celltype == CELL_INTERIOR) | (celltype == CELL_WALL)
        if gamma_vof[interior].mean() > 0.985:
            take_snapshot(step, t)
            break
    return snapshots
