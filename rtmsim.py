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

Object model
------------
    ResinMaterial    resin flow / thermal / cure properties
    FabricMaterial   dry reinforcement: permeability, porosity,
                     compaction law, fibre thermal properties
    LaminateStack    ordered plies (fabric + thickness + fibre direction,
                     global or per element); one default stack plus
                     optional per-region stacks (PCOMP-style)
    ShellMesh        triangle shell mesh backed by a trimesh.Trimesh
                     (STL file, trimesh object or numpy arrays)
    SolverSettings   numerical constants (CFL, EOS fit, dt control, ...)
    RTMSimulation    configured through setters, run with run(), results
                     exposed through getters

Physical inputs have no built-in defaults: every value the selected
model needs must be set explicitly, and RTMSimulation.validate() names
whatever is missing. Numerical settings default to the values the
solver has always used.

Minimal example::

    mesh = ShellMesh.from_stl("part.stl", scale=1e-3, units="mm")
    resin = ResinMaterial("epoxy").set_viscosity(0.1)
    fabric = (FabricMaterial("biax").set_permeability(3e-10, 6e-11)
              .set_porosity(0.6))
    stack = LaminateStack().add_ply(fabric, 0.75e-3, refdir=(1, 0, 0))
    sim = (RTMSimulation()
           .set_mesh(mesh).set_process_model(1)
           .set_resin(resin).set_laminate(stack)
           .set_pressures(p_inlet=2e5, p_init=1e5)
           .set_air_eos(p_ref=1.01325e5, rho_ref=1.225, gamma=1.4)
           .set_run_control(tmax=600.0, n_pics=16)
           .add_injection_port((10.0, 0.0, 5.0), radius=5.0)
           .add_injection_port((200.0, 0.0, 5.0), radius=5.0,
                               t_activate=120.0))
    sim.run()
    t_fill = sim.get_total_fill_time()
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import warnings
import numpy as np
import trimesh


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
CELL_INTERIOR = 1
CELL_INLET = -1
CELL_OUTLET = -2
CELL_WALL = -3


def _positive(value, name):
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0 (got {value})")
    return value


def _non_negative(value, name):
    value = float(value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be >= 0 (got {value})")
    return value


def _missing(owner, names):
    if names:
        raise ValueError(f"{owner}: missing input(s): {', '.join(names)}")


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
class ResinMaterial:
    """
    Resin properties.

    Iso-thermal runs need the constant viscosity (plus the density for
    i_model 2/3). Thermal runs replace the constant viscosity by

        mu(T, alpha) = mu_inf * exp(E_mu / (R T))
                       * (alpha_gel / (alpha_gel - alpha))^(C1 + C2*alpha)

    capped at mu_max (the gel factor only when cure is enabled), and need
    the density and specific heat. Cure runs also need the Kamal-Sourour
    kinetics

        dalpha/dt = (A1 exp(-E1/RT) + A2 exp(-E2/RT) alpha^m) (1-alpha)^n

    with total heat of reaction H_total [J/kg].
    """

    def __init__(self, name="resin"):
        self._name = str(name)
        self._viscosity = None
        self._density = None
        self._cp = None
        self._conductivity = None
        self._mu_inf = None
        self._E_mu = None
        self._mu_max = None
        self._alpha_gel = None
        self._C1 = None
        self._C2 = None
        self._cure = None

    def __repr__(self):
        return f"ResinMaterial({self._name!r})"

    def get_name(self):
        return self._name

    def set_viscosity(self, mu):
        """Constant viscosity [Pa s] used by iso-thermal runs."""
        self._viscosity = _positive(mu, "viscosity")
        return self

    def get_viscosity(self):
        return self._viscosity

    def set_density(self, rho):
        """Resin density [kg/m^3]."""
        self._density = _positive(rho, "density")
        return self

    def get_density(self):
        return self._density

    def set_thermal_properties(self, cp, conductivity=None):
        """Specific heat [J/(kg K)] and thermal conductivity [W/(m K)]."""
        self._cp = _positive(cp, "cp")
        self._conductivity = (None if conductivity is None
                              else _positive(conductivity, "conductivity"))
        return self

    def get_specific_heat(self):
        return self._cp

    def get_conductivity(self):
        return self._conductivity

    def set_viscosity_model(self, mu_inf, E_mu, mu_max):
        """Arrhenius viscosity mu_inf*exp(E_mu/RT) [Pa s, J/mol], capped at mu_max."""
        self._mu_inf = _positive(mu_inf, "mu_inf")
        self._E_mu = _non_negative(E_mu, "E_mu")
        self._mu_max = _positive(mu_max, "mu_max")
        return self

    def set_gel_model(self, alpha_gel, C1, C2):
        """Castro-Macosko gel factor (alpha_gel/(alpha_gel-alpha))^(C1+C2*alpha)."""
        alpha_gel = float(alpha_gel)
        if not 0.0 < alpha_gel <= 1.0:
            raise ValueError("alpha_gel must be in (0, 1]")
        self._alpha_gel = alpha_gel
        self._C1 = float(C1)
        self._C2 = float(C2)
        return self

    def get_viscosity_model(self):
        return dict(mu_inf=self._mu_inf, E_mu=self._E_mu, mu_max=self._mu_max,
                    alpha_gel=self._alpha_gel, C1=self._C1, C2=self._C2)

    def set_cure_kinetics(self, H_total, A1, A2, E1, E2, m, n, alpha_init=0.0):
        """Kamal-Sourour kinetics [J/kg, 1/s, 1/s, J/mol, J/mol, -, -]."""
        self._cure = dict(
            H_total=_non_negative(H_total, "H_total"),
            A1=_non_negative(A1, "A1"), A2=_non_negative(A2, "A2"),
            E1=_non_negative(E1, "E1"), E2=_non_negative(E2, "E2"),
            m=_non_negative(m, "m"), n=_non_negative(n, "n"),
            alpha_init=_non_negative(alpha_init, "alpha_init"),
        )
        return self

    def get_cure_kinetics(self):
        return None if self._cure is None else dict(self._cure)

    def viscosity_at(self, T, alpha=None, mu_floor=1e-6):
        """Viscosity [Pa s] from the thermal model at temperature(s) T [K]."""
        _missing(repr(self), [k for k, v in (("mu_inf", self._mu_inf),
                                             ("E_mu", self._E_mu),
                                             ("mu_max", self._mu_max))
                              if v is None])
        T = np.atleast_1d(np.asarray(T, dtype=np.float64))
        cure_on = alpha is not None
        if cure_on:
            _missing(repr(self), [] if self._alpha_gel is not None
                     else ["gel model"])
        a = (np.broadcast_to(np.asarray(alpha, dtype=np.float64), T.shape).copy()
             if cure_on else np.zeros_like(T))
        return _viscosity_TA(T, a, self._mu_inf, self._E_mu,
                             self._alpha_gel if cure_on else 1.0,
                             self._C1 if cure_on else 0.0,
                             self._C2 if cure_on else 0.0,
                             self._mu_max, mu_floor, cure_on)

    def validate(self, i_model, thermal=False, cure=False):
        missing = []
        if not thermal and self._viscosity is None:
            missing.append("viscosity")
        if (i_model in (2, 3) or thermal) and self._density is None:
            missing.append("density")
        if thermal:
            if self._cp is None:
                missing.append("thermal properties (cp)")
            if self._mu_inf is None:
                missing.append("viscosity model (mu_inf, E_mu, mu_max)")
        if cure:
            if self._alpha_gel is None:
                missing.append("gel model (alpha_gel, C1, C2)")
            if self._cure is None:
                missing.append("cure kinetics")
        _missing(repr(self), missing)
        if cure and not self._cure["alpha_init"] < self._alpha_gel:
            raise ValueError(f"{self!r}: need alpha_init < alpha_gel")


class FabricMaterial:
    """
    Dry reinforcement (fabric) properties.

    K1 / K2 are the in-plane principal permeabilities [m^2] along / across
    the ply's fibre direction; porosity is the fibre-bed porosity. For
    i_model 3 the optional compaction law

        phi(p) = porosity + c * p^2,   c = (porosity_at_p1 - porosity) / p1^2

    makes porosity pressure dependent; without set_compaction() c = 0.
    porosity_at_p1 is the measured porosity at pressure p1 (typically the
    injection pressure). Thermal runs need the fibre density and specific
    heat.
    """

    def __init__(self, name="fabric"):
        self._name = str(name)
        self._K1 = None
        self._K2 = None
        self._porosity = None
        self._compaction = None
        self._density = None
        self._cp = None
        self._conductivity = None

    def __repr__(self):
        return f"FabricMaterial({self._name!r})"

    def get_name(self):
        return self._name

    def set_permeability(self, K1, K2):
        """Principal permeabilities [m^2] along (K1) and across (K2) the fibres."""
        self._K1 = _positive(K1, "K1")
        self._K2 = _positive(K2, "K2")
        return self

    def get_permeability(self):
        return self._K1, self._K2

    def set_porosity(self, porosity):
        porosity = float(porosity)
        if not 0.0 < porosity < 1.0:
            raise ValueError("porosity must be in (0, 1)")
        self._porosity = porosity
        return self

    def get_porosity(self):
        return self._porosity

    def set_compaction(self, porosity_at_p1, p1):
        """Quadratic compaction law through (p=0, porosity), (p1, porosity_at_p1)."""
        porosity_at_p1 = float(porosity_at_p1)
        if not 0.0 < porosity_at_p1 < 1.0:
            raise ValueError("porosity_at_p1 must be in (0, 1)")
        self._compaction = (porosity_at_p1, _positive(p1, "p1"))
        return self

    def clear_compaction(self):
        self._compaction = None
        return self

    def get_compaction(self):
        """(porosity_at_p1, p1), or None when the preform is rigid."""
        return self._compaction

    def get_porosity_quadratic_c(self):
        if self._compaction is None:
            return 0.0
        porosity_at_p1, p1 = self._compaction
        return (porosity_at_p1 - self._porosity) / (p1 * p1)

    def set_thermal_properties(self, density, cp, conductivity=None):
        """Fibre density [kg/m^3], specific heat [J/(kg K)], conductivity [W/(m K)]."""
        self._density = _positive(density, "fibre density")
        self._cp = _positive(cp, "fibre cp")
        self._conductivity = (None if conductivity is None
                              else _positive(conductivity, "conductivity"))
        return self

    def get_density(self):
        return self._density

    def get_specific_heat(self):
        return self._cp

    def get_conductivity(self):
        return self._conductivity

    def validate(self, thermal=False):
        missing = []
        if self._K1 is None:
            missing.append("permeability (K1, K2)")
        if self._porosity is None:
            missing.append("porosity")
        if thermal and self._density is None:
            missing.append("thermal properties (density, cp)")
        _missing(repr(self), missing)


class Ply:
    """
    One ply: a fabric, its thickness [m] and fibre direction.

    refdir is a global-frame vector (3,) shared by every element, or one
    vector per element (N, 3) (e.g. from a draping tool or
    ShellMesh.get_cylindrical_directions). It is projected onto each
    element's tangent plane. angle_deg [deg] then rotates the projected
    direction about the element normal (right-hand rule on the mesh face
    orientation); it is a scalar or one value per element (N,), e.g. a
    local fibre deviation.
    """

    def __init__(self, fabric, thickness, refdir, angle_deg=0.0):
        if not isinstance(fabric, FabricMaterial):
            raise TypeError("fabric must be a FabricMaterial")
        refdir = np.array(refdir, dtype=np.float64)
        if refdir.ndim == 1:
            refdir = refdir.reshape(3)
        elif refdir.ndim != 2 or refdir.shape[1] != 3:
            raise ValueError("refdir must be a (3,) or (N, 3) array")
        if not np.all(np.linalg.norm(refdir, axis=-1) > 0.0):
            raise ValueError("refdir must be non-zero")
        angle_deg = np.array(angle_deg, dtype=np.float64)
        if angle_deg.ndim > 1:
            raise ValueError("angle_deg must be a scalar or an (N,) array")
        if not np.all(np.isfinite(angle_deg)):
            raise ValueError("angle_deg must be finite")
        self._fabric = fabric
        self._thickness = _positive(thickness, "ply thickness")
        self._refdir = refdir
        self._angle_deg = angle_deg

    def __repr__(self):
        refdir = (self._refdir.tolist() if self._refdir.ndim == 1
                  else f"per-element({self._refdir.shape[0]})")
        angle = (f", angle_deg={float(self._angle_deg):g}"
                 if self._angle_deg.ndim == 0 and self._angle_deg != 0.0
                 else (f", angle_deg=per-element({self._angle_deg.size})"
                       if self._angle_deg.ndim == 1 else ""))
        return (f"Ply({self._fabric.get_name()!r}, "
                f"t={self._thickness:g}, refdir={refdir}{angle})")

    def get_fabric(self):
        return self._fabric

    def get_thickness(self):
        return self._thickness

    def get_refdir(self):
        """(3,) global direction or (N, 3) per-element directions."""
        return self._refdir.copy()

    def get_angle_deg(self):
        """Scalar or (N,) rotation about the element normal [deg]."""
        return self._angle_deg.copy()

    def validate(self, n_cells):
        """Per-element refdir / angle arrays must have one row per cell."""
        if self._refdir.ndim == 2 and self._refdir.shape[0] != n_cells:
            raise ValueError(f"{self!r}: refdir has {self._refdir.shape[0]} "
                             f"rows, mesh has {n_cells} cells")
        if self._angle_deg.ndim == 1 and self._angle_deg.size != n_cells:
            raise ValueError(f"{self!r}: angle_deg has {self._angle_deg.size} "
                             f"values, mesh has {n_cells} cells")


class LaminateStack:
    """
    Ordered stack of plies (PCOMP-style). A simulation uses one default
    stack for every element; RTMSimulation.add_laminate_region assigns
    other stacks (different ply count / thickness / fabrics, e.g. ply
    drops or UD reinforcements) to subsets of elements. The element's
    geometric (e1, e2, e3) frame is built from
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

    def __init__(self):
        self._plies = []

    def __repr__(self):
        return f"LaminateStack({len(self._plies)} plies)"

    def add_ply(self, fabric, thickness, refdir, angle_deg=0.0):
        """Append a ply; see Ply for refdir / angle_deg conventions."""
        self._plies.append(Ply(fabric, thickness, refdir, angle_deg))
        return self

    def get_plies(self):
        return tuple(self._plies)

    def get_num_plies(self):
        return len(self._plies)

    def get_total_thickness(self):
        return float(sum(p.get_thickness() for p in self._plies))

    def get_effective_porosity(self):
        t = self.get_total_thickness()
        if t == 0:
            return 0.0
        return float(sum(p.get_fabric().get_porosity() * p.get_thickness()
                         for p in self._plies) / t)

    def get_effective_c_porosity(self):
        """Thickness-weighted c-coefficient for phi_eff(p) = phi0 + c*p^2."""
        t = self.get_total_thickness()
        if t == 0:
            return 0.0
        return float(sum(p.get_fabric().get_porosity_quadratic_c()
                         * p.get_thickness() for p in self._plies) / t)

    def get_effective_fibre_property(self, getter):
        """Thickness-weighted fabric property, e.g. FabricMaterial.get_density."""
        vals = [getter(p.get_fabric()) for p in self._plies]
        if all(v == vals[0] for v in vals):
            return float(vals[0])
        return float(sum(v * p.get_thickness() for v, p in zip(vals, self._plies))
                     / self.get_total_thickness())

    def validate(self, thermal=False, n_cells=None):
        if not self._plies:
            raise ValueError("LaminateStack has no plies (use add_ply)")
        if n_cells is not None:
            for ply in self._plies:
                ply.validate(n_cells)
        for fabric in {id(p.get_fabric()): p.get_fabric()
                       for p in self._plies}.values():
            fabric.validate(thermal=thermal)


# --------------------------------------------------------------------------
# Numerical settings
# --------------------------------------------------------------------------
class SolverSettings:
    """
    Numerical constants of the solver. Defaults reproduce the historical
    behaviour; change them with set(name=value, ...) and read them with
    get(name) or as attributes (settings.cfl).

    Time stepping
      cfl                    Courant number on the Darcy velocity
      max_neighbours         neighbour slots per cell
      min_steps_per_snapshot dt cap: at least this many steps per snapshot
      dt_growth_rtm          adaptive dt may grow to this x initial dt (i_model 1)
      dt_growth_two_fluid    same for the stiffer two-fluid models (2, 3)
      dt_adapt_after_steps   steps at the initial dt before adapting
      h_min_mode             "min": CFL length from the smallest cell;
                             "percentile": cells smaller than the
                             h_min_percentile-th area percentile are clamped
                             to it, so a few sliver triangles cannot collapse dt
      h_min_percentile       area percentile [%] for h_min_mode="percentile"
      dt_thermal_safety      factor on the Newton-cooling dt limit
    EOS
      p_eps                  pressure origin of the normalised i_model 1 run [Pa]
      eos_fit_pressures      3 pressures [Pa] of the quadratic p(rho) fit (i_model 1)
      exp_eos                power-law exponent (i_model 2/3); 0 = automatic
      exp_eos_default        automatic exponent for normal preforms
      exp_eos_racetrack      automatic exponent when anisotropy >= racetrack_perm_ratio
      racetrack_perm_ratio   K_max/K_min ratio that triggers race-tracking mode
      racetrack_dt_factor    dt factor in race-tracking mode
    Compaction (i_model 3)
      porosity_relaxation    relaxation factor of the effective porosity per step
      porosity_clip          (min, max) porosity during the run
      porosity_clip_init     (min, max) porosity at initialisation
      max_porosity_at_inlet  validation limit on phi(p_inlet)
    Thermal / cure
      mu_floor               lower viscosity bound [Pa s]
      max_dalpha_per_step    cap on the conversion increment per step
    Parallel execution
      n_threads              CPU threads for the flow kernel; None = automatic
                             (about cells_per_thread cells per thread, capped
                             at the machine's thread count). Small meshes run
                             faster on few threads: waking threads costs more
                             than the work. Results do not depend on it.
      cells_per_thread       cell count per thread for n_threads=None
    Termination / post-processing
      fill_stop_fraction     stop when the mean fill fraction exceeds this
      fill_time_threshold    gamma at which a cell counts as filled (fill-time field)
    """

    _DEFAULTS = dict(
        cfl=0.05,
        max_neighbours=10,
        min_steps_per_snapshot=40,
        dt_growth_rtm=1000.0,
        dt_growth_two_fluid=50.0,
        dt_adapt_after_steps=4,
        h_min_mode="min",
        h_min_percentile=1.0,
        dt_thermal_safety=0.5,
        p_eps=100.0,
        eos_fit_pressures=(0.0, 0.5e5, 1.0e5),
        exp_eos=0,
        exp_eos_default=4,
        exp_eos_racetrack=25,
        racetrack_perm_ratio=100.0,
        racetrack_dt_factor=0.1,
        porosity_relaxation=0.01,
        porosity_clip=(1e-3, 0.95),
        porosity_clip_init=(1e-6, 0.999),
        max_porosity_at_inlet=0.9,
        mu_floor=1e-6,
        max_dalpha_per_step=0.05,
        fill_stop_fraction=0.985,
        fill_time_threshold=0.5,
        n_threads=None,
        cells_per_thread=500,
    )

    def __init__(self, **overrides):
        self._values = dict(self._DEFAULTS)
        self.set(**overrides)

    def __repr__(self):
        changed = {k: v for k, v in self._values.items()
                   if v != self._DEFAULTS[k]}
        return f"SolverSettings({changed})"

    def __getattr__(self, name):
        try:
            return self.__dict__["_values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._DEFAULTS:
                raise KeyError(f"Unknown solver setting {k!r}; "
                               f"valid: {sorted(self._DEFAULTS)}")
            if k == "h_min_mode" and v not in ("min", "percentile"):
                raise ValueError("h_min_mode must be 'min' or 'percentile'")
            self._values[k] = v
        return self

    def get(self, name):
        if name not in self._values:
            raise KeyError(name)
        return self._values[name]

    def as_dict(self):
        return dict(self._values)


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------
class ShellMesh:
    """
    Triangle shell mesh, backed by a trimesh.Trimesh in solver units (m).

    `scale` converts the input units (e.g. mm for an STL) to metres;
    coordinates passed to port setters and returned for display are in
    input units. `cellgridid` is the row-sorted connectivity the solver
    works with; `faces` keeps the original orientation for rendering.
    """

    def __init__(self, nodes, faces, scale=1.0, units="m"):
        nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must be an (N, 3) triangle array")
        self._scale = _positive(scale, "scale")
        self._units = str(units)
        self.nodes = nodes
        self.faces = faces
        self.cellgridid = np.sort(faces, axis=1)
        self.cellcenter = nodes[self.cellgridid].mean(axis=1)
        self.N = faces.shape[0]
        self._tm = trimesh.Trimesh(vertices=nodes, faces=faces, process=False)
        self._body_ids = None

    def __repr__(self):
        return (f"ShellMesh({self.N} cells, {self.nodes.shape[0]} nodes, "
                f"units={self._units!r}, scale={self._scale:g})")

    # ---- constructors ----
    @classmethod
    def from_arrays(cls, nodes, triangles, scale=1.0, units="m"):
        """Nodes in input units, triangles as (N, 3) node indices."""
        nodes = np.asarray(nodes, dtype=np.float64)
        if scale != 1.0:
            nodes = nodes * scale
        return cls(nodes, triangles, scale=scale, units=units)

    @classmethod
    def from_trimesh(cls, tm, scale=1.0, units="m", keep_largest_body=False):
        """Copy a trimesh.Trimesh (vertices in input units)."""
        if keep_largest_body:
            bodies = tm.split(only_watertight=False)
            tm = max(bodies, key=lambda b: len(b.faces))
        return cls.from_arrays(np.array(tm.vertices), np.array(tm.faces),
                               scale=scale, units=units)

    @classmethod
    def from_stl(cls, path, scale=1.0, units="m", keep_largest_body=False):
        """Load an STL (any trimesh-readable surface); duplicate vertices are merged."""
        tm = trimesh.load(path, force="mesh", process=True)
        return cls.from_trimesh(tm, scale=scale, units=units,
                                keep_largest_body=keep_largest_body)

    @classmethod
    def make_square_plate(cls, side, n_div, z=0.0):
        """Structured triangle mesh of a square plate centred at the origin [m]."""
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
        return cls(nodes, np.array(tris, dtype=np.int64))

    # ---- getters ----
    def get_trimesh(self):
        return self._tm

    def get_num_cells(self):
        return self.N

    def get_scale(self):
        return self._scale

    def get_units(self):
        return self._units

    def get_cell_areas(self):
        """Cell areas [m^2]."""
        return self._tm.area_faces.copy()

    def get_cell_normals(self):
        """Unit cell normals from the original face orientation."""
        return self._tm.face_normals.copy()

    def get_cylindrical_directions(self, origin, axis, kind="hoop"):
        """
        Per-cell fibre directions (N, 3) of a cylindrical frame, for use
        as a per-element Ply refdir. `origin` (input units) and `axis`
        define the cylinder axis; kind is "hoop" (axis x radial),
        "radial" or "axial". Cells on the axis get a radial direction of
        zero length and are rejected.
        """
        a = np.asarray(axis, dtype=np.float64).reshape(3)
        if not np.linalg.norm(a) > 0.0:
            raise ValueError("axis must be a non-zero vector")
        a = a / np.linalg.norm(a)
        if kind == "axial":
            return np.tile(a, (self.N, 1))
        d = self.cellcenter - self.to_solver_units(
            np.asarray(origin, dtype=np.float64).reshape(3))
        radial = d - (d @ a)[:, None] * a
        r = np.linalg.norm(radial, axis=1)
        if np.any(r <= 1e-12 * max(float(r.max()), 1e-300)):
            raise ValueError("cells lie on the cylinder axis")
        radial /= r[:, None]
        if kind == "radial":
            return radial
        if kind == "hoop":
            return np.cross(a, radial)
        raise ValueError('kind must be "hoop", "radial" or "axial"')

    def to_solver_units(self, x):
        return np.asarray(x, dtype=np.float64) * self._scale

    def to_input_units(self, x):
        return np.asarray(x, dtype=np.float64) / self._scale

    def get_body_ids(self):
        """Connected-component label per cell (edge connectivity)."""
        if self._body_ids is None:
            self._body_ids = trimesh.graph.connected_component_labels(
                self._tm.face_adjacency, node_count=self.N).astype(np.int64)
        return self._body_ids

    def get_bodies(self):
        """List of cell-index arrays, one per connected body, largest first."""
        ids = self.get_body_ids()
        bodies = [np.where(ids == b)[0] for b in np.unique(ids)]
        return sorted(bodies, key=len, reverse=True)

    def get_quality_report(self, sliver_ratio=0.01):
        """Mesh statistics; slivers are cells below sliver_ratio x median area."""
        area = self._tm.area_faces / self._scale ** 2
        edges = self.cellgridid[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2)
        n_boundary = len(trimesh.grouping.group_rows(edges, require_count=1))
        med = float(np.median(area))
        return dict(
            n_cells=self.N, n_nodes=int(self.nodes.shape[0]),
            n_bodies=len(self.get_bodies()),
            body_sizes=[len(b) for b in self.get_bodies()],
            area_min=float(area.min()), area_median=med,
            area_max=float(area.max()),
            n_slivers=int((area < sliver_ratio * med).sum()),
            n_boundary_edges=n_boundary,
            bounds=self.to_input_units(self._tm.bounds),
            units=self._units,
        )

    # ---- point queries (input units) ----
    def find_cells_near_point(self, point, radius=0.0, snap_to_surface=True,
                              close_pockets=True):
        """
        Cells whose centre lies within `radius` of `point` (input units).

        snap_to_surface=True first projects the point onto the surface and
        falls back to the cell containing the projection when no centre is
        within `radius`; otherwise the raw point is used and the fallback
        is the closest cell centre. close_pockets=True then absorbs cells
        enclosed on two or more edges by the selection (see
        close_pockets()). Returns (cells, center, distance), with
        `center` the search centre and `distance` the point-to-surface (or
        point-to-closest-centre) distance, both in input units.
        """
        p = self.to_solver_units(np.asarray(point, dtype=np.float64).reshape(3))
        r = _non_negative(radius, "radius") * self._scale
        if snap_to_surface:
            closest, dist, tri = trimesh.proximity.closest_point(
                self._tm, p[None, :])
            center = closest[0]
            fallback = int(tri[0])
            d0 = float(dist[0])
            d = np.linalg.norm(self.cellcenter - center, axis=1)
        else:
            center = p
            d = np.linalg.norm(self.cellcenter - center, axis=1)
            fallback = int(np.argmin(d))
            d0 = float(d[fallback])
        within = np.where(d <= r)[0]
        cells = (within.astype(int) if within.size
                 else np.array([fallback], dtype=int))
        if close_pockets:
            cells = self.close_pockets(cells)
        return cells, self.to_input_units(center), d0 / self._scale

    def close_pockets(self, cells):
        """
        Grow a cell set until no outside cell shares two or more edges
        with it. A cell wedged between port cells gets inflow through
        several inlet faces from its own pressure gradient and can build
        up pressure beyond the inlet value, so such pockets are absorbed.
        """
        adj = self._tm.face_adjacency
        sel = np.zeros(self.N, dtype=bool)
        sel[np.asarray(cells, dtype=int)] = True
        while True:
            count = np.zeros(self.N, dtype=np.int64)
            np.add.at(count, adj[:, 0], sel[adj[:, 1]])
            np.add.at(count, adj[:, 1], sel[adj[:, 0]])
            pocket = ~sel & (count >= 2)
            if not pocket.any():
                return np.where(sel)[0].astype(int)
            sel |= pocket

    def find_cells_by_predicate(self, predicate):
        """Cells for which predicate(cellcenters_in_input_units) is True."""
        mask = np.asarray(predicate(self.to_input_units(self.cellcenter)),
                          dtype=bool)
        return np.where(mask)[0].astype(int)

    # ---- display ----
    def to_pyvista(self, cell_data=None, input_units=True):
        """pyvista.PolyData of the surface (original face orientation)."""
        import pyvista as pv
        pts = self.to_input_units(self.nodes) if input_units else self.nodes
        cells = np.column_stack([np.full(self.N, 3, dtype=np.int64),
                                 self.faces]).ravel()
        poly = pv.PolyData(pts, cells)
        poly.cell_data["body"] = self.get_body_ids()
        for name, values in (cell_data or {}).items():
            poly.cell_data[name] = np.asarray(values)
        return poly


# --------------------------------------------------------------------------
# Ports and results
# --------------------------------------------------------------------------
@dataclass
class InjectionPort:
    """
    Injection port (kind="inlet") or vent (kind="vent").

    Located either by explicit cell ids or by a point + radius in mesh
    input units (see ShellMesh.find_cells_near_point). An inlet with
    t_activate == 0 is active from the start; t_activate > 0 makes it a
    cascade (delayed) port that switches to inlet at that time and stays
    pinned at inlet conditions for the rest of the run.
    """
    name: str
    kind: str
    t_activate: float
    coords: Optional[np.ndarray] = None
    radius: float = 0.0
    snap_to_surface: bool = True
    cell_ids: Optional[np.ndarray] = None
    # Filled in by RTMSimulation when resolved against the mesh
    cells: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None      # input units
    snap_distance: Optional[float] = None    # input units

    @property
    def is_cascade(self):
        return self.kind == "inlet" and self.t_activate > 0.0


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

    def get_pressure_absolute(self):
        """Per-cell absolute pressure [Pa] for any model."""
        return self.p + self.p_offset

    def get_fluid_mask(self):
        """Interior + wall cells (excludes imposed inlet/outlet cells)."""
        return (self.celltype == CELL_INTERIOR) | (self.celltype == CELL_WALL)


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


# --------------------------------------------------------------------------
# Topology + local coord systems (port of tools.jl)
# --------------------------------------------------------------------------
def create_faces(mesh, max_neighbours):
    """
    Edge adjacency. Returns (neighbours, celltype).

    Edges are grouped with trimesh; neighbours are filled in the legacy
    traversal order (cells in order, edges (n0,n1), (n1,n2), (n0,n2) of
    the sorted triangle), so results match the original solver exactly.
    Cells with an unshared edge are walls.
    """
    N = mesh.N
    edges = mesh.cellgridid[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2)
    groups = sorted((np.sort(g) for g in trimesh.grouping.group_rows(edges)),
                    key=lambda g: g[0])
    neighbours = np.full((N, max_neighbours), -9, dtype=np.int64)
    celltype = np.full(N, CELL_INTERIOR, dtype=np.int64)
    fill = np.zeros(N, dtype=np.int64)
    for rows in groups:
        cells = rows // 3
        if cells.size == 1:
            celltype[cells[0]] = CELL_WALL
            continue
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


def element_frames(mesh):
    """
    Per-element geometric frame (e1, e2, e3), each (N, 3), plus the sign
    (N,) of e3 against the original face normal. e1 runs along the first
    edge of the sorted triangle, e2 is in-plane orthogonal to it; this is
    the frame the solver and the K tensor work in.
    """
    nodes = mesh.nodes
    cg = mesh.cellgridid
    e1 = nodes[cg[:, 1]] - nodes[cg[:, 0]]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    a2 = nodes[cg[:, 2]] - nodes[cg[:, 0]]
    a2 /= np.linalg.norm(a2, axis=1)[:, None]
    e2 = a2 - np.einsum("ij,ij->i", e1, a2)[:, None] * e1
    e2 /= np.linalg.norm(e2, axis=1)[:, None]
    e3 = np.cross(e1, e2)
    f = mesh.faces
    n_face = np.cross(nodes[f[:, 1]] - nodes[f[:, 0]],
                      nodes[f[:, 2]] - nodes[f[:, 0]])
    sign = np.where(np.einsum("ij,ij->i", e3, n_face) < 0.0, -1.0, 1.0)
    return e1, e2, e3, sign


def build_stack_tensor(mesh, stacks, stack_id, deviation_deg=None,
                       normal_tol=0.1):
    """
    Per-element 2x2 in-plane permeability tensor for stacked laminates.

    stacks is a list of LaminateStack, stack_id (N,) the stack index of
    each element (one stack per element, plies may differ between
    stacks). For each element and each ply of its stack, the ply refdir
    (global or per-element) is projected onto the element tangent plane
    and normalised; the ply angle_deg plus the element deviation_deg
    (None, scalar or (N,)) then rotate it about the face normal. The
    ply's tensor in the element frame (e1, e2) is

        R(alpha_p) . diag(K1_p, K2_p) . R(alpha_p)^T

    and the element tensor is the thickness-weighted sum of ply tensors
    divided by the thickness of that element's stack (parallel flow
    through the plies).

    A refdir closer than normal_tol (in-plane component) to the element
    normal gives an ill-defined fibre angle and triggers a warning; an
    exactly normal refdir falls back to e1.

    Returns (Kxx, Kxy, Kyy), each (N,).
    """
    N = mesh.N
    stack_id = np.asarray(stack_id, dtype=np.int64)
    e1, e2, _, sign = element_frames(mesh)
    Kxx = np.zeros(N)
    Kxy = np.zeros(N)
    Kyy = np.zeros(N)
    dev = (np.zeros(N) if deviation_deg is None
           else np.broadcast_to(np.asarray(deviation_deg, dtype=np.float64),
                                (N,)))
    n_steep = 0

    for s, stack in enumerate(stacks):
        cells = np.where(stack_id == s)[0]
        if cells.size == 0:
            continue
        t_tot = stack.get_total_thickness()
        if t_tot <= 0:
            raise ValueError(f"{stack!r} has zero total thickness")
        steep = np.zeros(cells.size, dtype=bool)
        for ply in stack.get_plies():
            K1p, K2p = ply.get_fabric().get_permeability()
            w = ply.get_thickness()
            r = ply.get_refdir()
            r = (np.broadcast_to(r, (cells.size, 3)) if r.ndim == 1
                 else r[cells])
            r = r / np.linalg.norm(r, axis=1, keepdims=True)
            rx = np.einsum("ij,ij->i", r, e1[cells])
            ry = np.einsum("ij,ij->i", r, e2[cells])
            mag = np.sqrt(rx * rx + ry * ry)
            steep |= mag < normal_tol
            flat = mag < 1e-30
            mag = np.where(flat, 1.0, mag)
            c = np.where(flat, 1.0, rx / mag)
            sn = np.where(flat, 0.0, ry / mag)

            ang = np.broadcast_to(ply.get_angle_deg(), (N,))[cells] + dev[cells]
            if np.any(ang != 0.0):
                # Rotate about the face normal; sign maps it onto e3.
                th = np.radians(ang) * sign[cells]
                ct, st_ = np.cos(th), np.sin(th)
                c, sn = c * ct - sn * st_, sn * ct + c * st_

            Kxx[cells] += (c * c * K1p + sn * sn * K2p) * w
            Kxy[cells] += (c * sn * (K1p - K2p)) * w
            Kyy[cells] += (sn * sn * K1p + c * c * K2p) * w

        Kxx[cells] /= t_tot
        Kxy[cells] /= t_tot
        Kyy[cells] /= t_tot
        n_steep += int(steep.sum())

    if n_steep:
        warnings.warn(
            f"{n_steep} element(s) have a ply refdir within "
            f"{np.degrees(np.arcsin(normal_tol)):.1f} deg of the element "
            f"normal: the projected fibre angle there is ill-defined. Use a "
            f"per-element refdir (e.g. ShellMesh.get_cylindrical_directions).",
            RuntimeWarning, stacklevel=2)
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
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    prange = range
    def njit(*args, **kwargs):
        # no-op decorator if numba is not installed
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(f): return f
        return wrap


@njit(cache=True, fastmath=True, parallel=True)
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

    Cells are updated independently from the old state (Jacobi style), so
    the cell loop runs in parallel and gives the same result as serial.
    """
    N = rho.size
    K = nbrs.shape[1]

    rho_new = np.empty(N)
    u_new = np.empty(N)
    v_new = np.empty(N)
    p_new = np.empty(N)
    g_new = np.empty(N)
    rho_thresh = 0.5 * rho_resin

    for i in prange(N):
        cti = ct[i]
        if cti != 1 and cti != -3:
            # Inlet / vent cells keep their pinned state.
            rho_new[i] = rho[i]
            p_new[i] = p[i]
            g_new[i] = 1.0 if cti == -1 else 0.0
            u_new[i] = 0.0
            v_new[i] = 0.0
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

    for i in prange(N):
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
                  C1, C2, mu_max, mu_floor, cure_on):
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
        if mu < mu_floor:
            mu = mu_floor
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


# Kept serial: a parallel build reorders the fastmath flux sum and changes
# results at round-off level, which the cure runs amplify.
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
    da_max,
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
        rcp_fiber = (1.0 - phi) * rho_fiber[i] * cp_fiber[i]
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
            if da > da_max:
                da = da_max
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


# Return codes of _advance_jit: why it handed control back to run().
ADV_TMAX = 0       # t > t_max: run over
ADV_CASCADE = 1    # a cascade port is due: activate it, then resume
ADV_EVENT = 2      # snapshot due and/or fill criterion met
ADV_DIVERGED = 3   # dt became non-finite


@njit(cache=True, parallel=True)
def _update_fill_time_jit(fill_time, gamma_vof, g_fill, t):
    """Stamp t on cells whose gamma first reaches g_fill."""
    for i in prange(fill_time.size):
        if np.isnan(fill_time[i]) and gamma_vof[i] >= g_fill:
            fill_time[i] = t


@njit(cache=True, parallel=True)
def _cfl_min_jit(ct, u, v, h_cfl):
    """
    min(h / |u|) over interior and wall cells; NaN if any velocity is NaN
    (so a diverged state is reported, as np.min would).
    """
    h_over_s = np.inf
    n_nan = 0
    for i in prange(ct.size):
        if ct[i] == 1 or ct[i] == -3:
            speed = np.sqrt(u[i] * u[i] + v[i] * v[i]) + 1e-12
            r = h_cfl[i] / speed
            if np.isnan(r):
                n_nan += 1
            else:
                h_over_s = min(h_over_s, r)
    if n_nan > 0:
        return np.nan
    return h_over_s


@njit(cache=True)
def _advance_jit(
    i_model, thermal_on, cure_on,
    ct, nbrs, volume, cc_to_cc_x, cc_to_cc_y,
    face_nx, face_ny, face_area,
    T11, T12, T21, T22,
    Kxx_base, Kxy_base, Kyy_base, perm_factor, viscosity,
    phi_old, phi_new, phi0, c_porosity, phi_lo, phi_hi, phi_relax,
    rho, u, v, p, gamma_vof,
    ap0, ap1, ap2,
    p_a_eos, p_init_eos, c_eos, exp_eos, rho_air, rho_resin,
    thickness, porosity, T_field, alpha_field,
    cp_resin, rho_fiber, cp_fiber, h_tool, T_tool,
    mu_inf, E_mu, alpha_gel, C1, C2, mu_max, mu_floor,
    H_total, A1, A2, E1, E2, m_kin, n_kin, da_max,
    fill_time, g_fill, fill_region, h_cfl,
    t, step, dt, dt0, dt_growth, dt_pic_max, cfl_fac, adapt_after,
    t_max, t_next, t_cascade, fill_stop,
):
    """
    Time loop of RTMSimulation.run, compiled. Advances the state in place
    until something needs Python (cascade activation, snapshot, end of
    run) and returns (code, t, step, dt, snapshot_due, fill_done).

    The per-step operations and their order are those of the former
    Python loop, written as scalar loops with the same arithmetic, so the
    results are unchanged. No fastmath here, to keep numpy semantics.
    """
    N = rho.size
    while t <= t_max:
        if t >= t_cascade:
            return ADV_CASCADE, t, step, dt, False, False

        # i_model=3: refresh porosity / permeability factor before the step
        if i_model == 3:
            for i in range(N):
                pi = p[i]
                phi_p = phi0[i] + c_porosity[i] * (pi * pi)
                phi_p = min(max(phi_p, phi_lo), phi_hi)
                one_m = 1.0 - phi_p
                perm_factor[i] = phi_p ** 3.0 / (one_m * one_m)
                phi_target = (1.0 - phi0[i]) / one_m * phi_p
                phi_new[i] = phi_old[i] + phi_relax * (phi_target - phi_old[i])

        # Viscosity from current (T, alpha) so Darcy drag sees it.
        if thermal_on:
            viscosity[:] = _viscosity_TA(
                T_field, alpha_field, mu_inf, E_mu, alpha_gel,
                C1, C2, mu_max, mu_floor, cure_on)

        _step_jit(
            i_model,
            ct, nbrs, volume, cc_to_cc_x, cc_to_cc_y,
            face_nx, face_ny, face_area,
            T11, T12, T21, T22,
            Kxx_base, Kxy_base, Kyy_base, perm_factor, viscosity,
            phi_old, phi_new,
            rho, u, v, p, gamma_vof,
            dt,
            ap0, ap1, ap2,
            p_a_eos, p_init_eos, c_eos, exp_eos, rho_air, rho_resin,
        )
        if i_model == 3:
            phi_old[:] = phi_new

        if thermal_on:
            _step_thermal_jit(
                ct, nbrs, volume,
                face_nx, face_ny, face_area,
                T11, T12, T21, T22,
                thickness, porosity, gamma_vof,
                rho_resin, cp_resin, rho_fiber, cp_fiber,
                h_tool, T_tool,
                cure_on, H_total,
                A1, A2, E1, E2, m_kin, n_kin,
                da_max,
                u, v, T_field, alpha_field,
                dt,
            )

        step += 1
        t += dt
        _update_fill_time_jit(fill_time, gamma_vof, g_fill, t)

        # Adaptive dt: convective CFL over interior/wall cells.
        if step > adapt_after:
            h_over_s = _cfl_min_jit(ct, u, v, h_cfl)
            dt_conv = cfl_fac * h_over_s
            if np.isnan(dt_conv):
                dt = dt_conv
            else:
                dt = min(max(dt_conv, dt0), dt_growth * dt0, dt_pic_max)
        if not np.isfinite(dt):
            return ADV_DIVERGED, t, step, dt, False, False

        snap_due = t >= t_next or t + dt > t_max
        g_sum = 0.0
        n_sum = 0
        for i in range(N):
            if (ct[i] == 1 or ct[i] == -3) and fill_region[i]:
                g_sum += gamma_vof[i]
                n_sum += 1
        fill_done = n_sum > 0 and g_sum / n_sum > fill_stop
        if snap_due or fill_done:
            return ADV_EVENT, t, step, dt, snap_due, fill_done
    return ADV_TMAX, t, step, dt, False, False


def _setup_eos_model1(p_ref, rho_ref, gamma, p_fit, p_a, p_init):
    """Compressible-air EOS lookup-table coefficients and inlet/init densities."""
    kappa = p_ref / (rho_ref ** gamma)
    p_int = np.asarray(p_fit, dtype=np.float64)
    rho_int = (p_int / kappa) ** (1.0 / gamma)
    A = np.column_stack([rho_int ** 2, rho_int, np.ones(3)])
    ap = np.linalg.solve(A, p_int)
    rho_a = (p_a / kappa) ** (1.0 / gamma)
    rho_init = (p_init / kappa) ** (1.0 / gamma)
    return ap, rho_a, rho_init


def _setup_eos_model23(p_a, p_init, rho_air, rho_resin, exp_eos):
    """Two-fluid surrogate p(rho) = p_init + c*(rho-rho_air)^exp_eos.

    p_a, p_init are the absolute pressures (not normalized).
    Returns (c_eos, rho_a, rho_init).
    """
    drho = rho_resin - rho_air
    c = (p_a - p_init) / (drho ** exp_eos)
    return c, rho_resin, rho_air


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


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
class RTMSimulation:
    """
    Filling simulation: configure with set_*/add_* (chainable), call
    run(), read results with get_*.

    Required inputs: mesh, process model, resin, laminate, pressures, air
    EOS, run control and at least one injection port active at t = 0.
    Thermal (set_thermal) and cure (enable_cure) are optional.

    Laminate: set_laminate() gives the default stack of every element;
    add_laminate_region() overrides it on a set of elements (later
    regions win), so elements can carry different stacking sequences
    and thicknesses. set_fibre_deviation() adds a per-element rotation
    [deg] to the fibre direction of every ply.
    """

    def __init__(self):
        self._mesh = None
        self._i_model = None
        self._resin = None
        self._stack = None
        self._laminate_regions = []
        self._fibre_deviation = None
        self._p_inlet = None
        self._p_init = None
        self._air = None
        self._tmax = None
        self._n_pics = None
        self._thermal = None
        self._cure_enabled = False
        self._settings = SolverSettings()
        self._ports = []
        self._clear_results()

    def _clear_results(self):
        self._snapshots = []
        self._fill_time = None
        self._fill_region = None
        self._fill_complete = False
        self._t_end = None

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------
    def set_mesh(self, mesh):
        if not isinstance(mesh, ShellMesh):
            raise TypeError("mesh must be a ShellMesh")
        self._mesh = mesh
        for port in self._ports:
            port.cells = None
        return self

    def set_process_model(self, i_model):
        """1 = RTM, 2 = RTM-VARI two-fluid, 3 = VARI compactable preform."""
        if i_model not in (1, 2, 3):
            raise ValueError("i_model must be 1, 2, or 3")
        self._i_model = int(i_model)
        return self

    def set_resin(self, resin):
        if not isinstance(resin, ResinMaterial):
            raise TypeError("resin must be a ResinMaterial")
        self._resin = resin
        return self

    def set_laminate(self, stack):
        """Default stack, used by every element not in a laminate region."""
        if stack is not None and not isinstance(stack, LaminateStack):
            raise TypeError("stack must be a LaminateStack")
        self._stack = stack
        return self

    def add_laminate_region(self, stack, cell_ids):
        """
        Assign `stack` to the elements `cell_ids` (e.g. a ply-drop zone or
        a BDF PID group). Regions are applied in order after the default
        stack, so a later region overrides an earlier one on shared cells.
        """
        if not isinstance(stack, LaminateStack):
            raise TypeError("stack must be a LaminateStack")
        cell_ids = np.unique(np.asarray(cell_ids, dtype=np.int64).ravel())
        if cell_ids.size == 0:
            raise ValueError("laminate region: empty cell list")
        self._laminate_regions.append((stack, cell_ids))
        return self

    def clear_laminate_regions(self):
        self._laminate_regions = []
        return self

    def set_fibre_deviation(self, angle_deg):
        """
        Per-element fibre deviation [deg] (scalar or (N,)), added to every
        ply's direction as a rotation about the face normal (right-hand
        rule on the mesh face orientation). None removes it.
        """
        if angle_deg is not None:
            angle_deg = np.array(angle_deg, dtype=np.float64)
            if angle_deg.ndim > 1 or not np.all(np.isfinite(angle_deg)):
                raise ValueError("fibre deviation must be a finite scalar "
                                 "or (N,) array")
        self._fibre_deviation = angle_deg
        return self

    def set_pressures(self, p_inlet, p_init):
        """Absolute injection and initial (vent / cavity) pressures [Pa]."""
        p_inlet = _positive(p_inlet, "p_inlet")
        p_init = _non_negative(p_init, "p_init")
        if p_inlet <= p_init:
            raise ValueError("p_inlet must be > p_init")
        self._p_inlet, self._p_init = p_inlet, p_init
        return self

    def set_air_eos(self, p_ref, rho_ref, gamma):
        """
        Air reference state: rho_ref [kg/m^3] at p_ref [Pa], adiabatic
        exponent gamma. i_model 1 uses the isentropic EOS; i_model 2/3 use
        rho_ref as the air density of the two-fluid surrogate.
        """
        self._air = (_positive(p_ref, "p_ref"), _positive(rho_ref, "rho_ref"),
                     _positive(gamma, "gamma"))
        return self

    def set_run_control(self, tmax, n_pics):
        """Maximum simulated time [s] and number of snapshots (rounded to x4)."""
        self._tmax = _positive(tmax, "tmax")
        n_pics = int(n_pics)
        if n_pics < 1:
            raise ValueError("n_pics must be >= 1")
        self._n_pics = n_pics
        return self

    def set_thermal(self, T_init, T_inlet, T_tool, h_tool):
        """
        Enable the lumped through-thickness thermal model: initial preform
        temperature, resin inlet temperature, tool temperature [K] and
        tool-resin heat transfer coefficient h_tool [W/(m^2 K)].
        """
        self._thermal = dict(
            T_init=_positive(T_init, "T_init (K)"),
            T_inlet=_positive(T_inlet, "T_inlet (K)"),
            T_tool=_positive(T_tool, "T_tool (K)"),
            h_tool=_non_negative(h_tool, "h_tool"),
        )
        return self

    def disable_thermal(self):
        self._thermal = None
        self._cure_enabled = False
        return self

    def enable_cure(self, enabled=True):
        """Cure kinetics + gel viscosity; requires set_thermal()."""
        self._cure_enabled = bool(enabled)
        return self

    def set_solver_settings(self, settings=None, **kwargs):
        """Replace the SolverSettings object and/or override single values."""
        if settings is not None:
            if not isinstance(settings, SolverSettings):
                raise TypeError("settings must be a SolverSettings")
            self._settings = settings
        self._settings.set(**kwargs)
        return self

    def add_injection_port(self, coords, t_activate=0.0, radius=0.0,
                           name=None, snap_to_surface=True):
        """
        Injection port at `coords` (mesh input units). Cells whose centre
        is within `radius` (input units) of the point projected onto the
        surface become inlet cells; with no cell in range, the cell under
        the projected point is used. t_activate > 0 makes it a cascade
        port that opens at that time [s].
        """
        return self._add_port("inlet", t_activate, name, coords=coords,
                              radius=radius, snap_to_surface=snap_to_surface)

    def add_injection_port_cells(self, cell_ids, t_activate=0.0, name=None):
        """Injection port on explicit cell ids (e.g. a BDF SET)."""
        return self._add_port("inlet", t_activate, name, cell_ids=cell_ids)

    def add_vent(self, coords, radius=0.0, name=None, snap_to_surface=True):
        """Vent (outlet held at p_init); same location rules as injection ports."""
        return self._add_port("vent", 0.0, name, coords=coords, radius=radius,
                              snap_to_surface=snap_to_surface)

    def add_vent_cells(self, cell_ids, name=None):
        return self._add_port("vent", 0.0, name, cell_ids=cell_ids)

    def clear_ports(self):
        self._ports = []
        return self

    def _add_port(self, kind, t_activate, name, coords=None, radius=0.0,
                  snap_to_surface=True, cell_ids=None):
        t_activate = _non_negative(t_activate, "t_activate")
        if name is None:
            prefix = "V" if kind == "vent" else ("C" if t_activate > 0 else "P")
            name = f"{prefix}{sum(p.kind == kind for p in self._ports) + 1}"
        if cell_ids is not None:
            cell_ids = np.unique(np.asarray(cell_ids, dtype=int).ravel())
            if cell_ids.size == 0:
                raise ValueError(f"port {name!r}: empty cell list")
        else:
            coords = np.asarray(coords, dtype=np.float64).reshape(3)
            radius = _non_negative(radius, "radius")
        self._ports.append(InjectionPort(
            name=str(name), kind=kind, t_activate=t_activate,
            coords=coords, radius=radius, snap_to_surface=bool(snap_to_surface),
            cell_ids=cell_ids))
        return self

    # ------------------------------------------------------------------
    # Input getters
    # ------------------------------------------------------------------
    def get_mesh(self):
        return self._mesh

    def get_process_model(self):
        return self._i_model

    def get_resin(self):
        return self._resin

    def get_laminate(self):
        """Default stack (see get_laminate_map for the per-element layout)."""
        return self._stack

    def get_laminate_regions(self):
        return list(self._laminate_regions)

    def get_fibre_deviation(self):
        return (None if self._fibre_deviation is None
                else self._fibre_deviation.copy())

    def get_laminate_map(self):
        """
        (stacks, stack_id): the distinct stacks in use and, per element,
        the index into `stacks`; -1 marks elements without a laminate.
        """
        if self._mesh is None:
            raise ValueError("RTMSimulation: set_mesh() first")
        N = self._mesh.N
        stacks = []
        stack_id = np.full(N, -1, dtype=np.int64)

        def index(st):
            for k, s in enumerate(stacks):
                if s is st:
                    return k
            stacks.append(st)
            return len(stacks) - 1

        if self._stack is not None:
            stack_id[:] = index(self._stack)
        for st, cells in self._laminate_regions:
            if cells.min() < 0 or cells.max() >= N:
                raise IndexError(f"laminate region {st!r}: cell id out of "
                                 f"range")
            stack_id[cells] = index(st)
        used = np.unique(stack_id[stack_id >= 0])
        if used.size < len(stacks):
            # Drop stacks fully overridden by later regions.
            remap = np.full(len(stacks), -1, dtype=np.int64)
            remap[used] = np.arange(used.size)
            stacks = [stacks[k] for k in used]
            stack_id = np.where(stack_id >= 0, remap[stack_id], -1)
        return stacks, stack_id

    def get_pressures(self):
        return self._p_inlet, self._p_init

    def get_air_eos(self):
        return self._air

    def get_run_control(self):
        return self._tmax, self._n_pics

    def get_thermal(self):
        return None if self._thermal is None else dict(self._thermal)

    def is_cure_enabled(self):
        return self._cure_enabled

    def get_solver_settings(self):
        return self._settings

    def get_ports(self):
        """Ports resolved against the mesh (cells, centre, snap distance)."""
        self._resolve_ports()
        return list(self._ports)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self):
        missing = [label for label, v in (
            ("mesh (set_mesh)", self._mesh),
            ("process model (set_process_model)", self._i_model),
            ("resin (set_resin)", self._resin),
            ("laminate (set_laminate / add_laminate_region)",
             self._stack if self._stack is not None
             else (self._laminate_regions or None)),
            ("pressures (set_pressures)", self._p_inlet),
            ("air EOS (set_air_eos)", self._air),
            ("run control (set_run_control)", self._tmax),
        ) if v is None]
        if not any(p.kind == "inlet" and p.t_activate == 0.0
                   for p in self._ports):
            missing.append("injection port active at t=0 (add_injection_port)")
        _missing("RTMSimulation", missing)
        thermal = self._thermal is not None
        if self._cure_enabled and not thermal:
            raise ValueError("enable_cure requires set_thermal()")
        self._resin.validate(self._i_model, thermal=thermal,
                             cure=self._cure_enabled)
        N = self._mesh.N
        stacks, stack_id = self.get_laminate_map()
        n_bare = int((stack_id < 0).sum())
        if n_bare:
            raise ValueError(f"{n_bare} cells have no laminate: set a default "
                             f"stack with set_laminate()")
        for st in stacks:
            st.validate(thermal=thermal, n_cells=N)
        dev = self._fibre_deviation
        if dev is not None and dev.ndim == 1 and dev.size != N:
            raise ValueError(f"fibre deviation has {dev.size} values, mesh "
                             f"has {N} cells")
        if self._i_model == 3:
            # Sanity-check the porosity quadratic at the operating
            # pressure: if phi(p_inlet) is close to 1 the preform is
            # essentially decompacted, the Carman-Kozeny-like factor
            # phi^3/(1-phi)^2 diverges (>= ~73 already at phi=0.9), and
            # dt has to collapse to keep the solver stable.
            lim = self._settings.max_porosity_at_inlet
            for st in stacks:
                for k, ply in enumerate(st.get_plies()):
                    fab = ply.get_fabric()
                    phi_at_inlet = (fab.get_porosity()
                                    + fab.get_porosity_quadratic_c()
                                    * self._p_inlet ** 2)
                    if phi_at_inlet > lim:
                        raise ValueError(
                            f"i_model=3: {st!r} ply {k} porosity at p_inlet="
                            f"{self._p_inlet:.0f} Pa is {phi_at_inlet:.3f} "
                            f"(> {lim}). Reduce porosity_at_p1 or increase p1.")
        self._resolve_ports()

    def _resolve_ports(self):
        if self._mesh is None:
            raise ValueError("RTMSimulation: set_mesh() before resolving ports")
        N = self._mesh.N
        for port in self._ports:
            if port.cells is not None:
                continue
            if port.cell_ids is not None:
                if port.cell_ids.min() < 0 or port.cell_ids.max() >= N:
                    raise IndexError(f"port {port.name!r}: cell id out of range")
                port.cells = port.cell_ids
                port.center = self._mesh.to_input_units(
                    self._mesh.cellcenter[port.cells].mean(axis=0))
                port.snap_distance = 0.0
            else:
                port.cells, port.center, port.snap_distance = \
                    self._mesh.find_cells_near_point(
                        port.coords, port.radius, port.snap_to_surface)

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def run(self, on_snapshot=None):
        """
        Run the filling simulation; returns the list of Snapshot objects
        and calls on_snapshot(snap) after each snapshot if provided.
        """
        self.validate()
        if not _HAVE_NUMBA:
            return self._run(on_snapshot)
        import numba
        st = self._settings
        n_max = numba.config.NUMBA_NUM_THREADS
        if st.n_threads is None:
            n_threads = self._mesh.N // max(int(st.cells_per_thread), 1)
        else:
            n_threads = int(st.n_threads)
        n_threads = min(max(n_threads, 1), n_max)
        previous = numba.get_num_threads()
        numba.set_num_threads(n_threads)
        try:
            return self._run(on_snapshot)
        finally:
            numba.set_num_threads(previous)

    def _run(self, on_snapshot):
        """
        Body of run().

        Three model branches share the same JIT kernel; per-cell arrays
        are precomputed here so the kernel sees a uniform interface:

          * Kxx_base, Kxy_base, Kyy_base : per-element K from build_stack_tensor
          * perm_factor[i]               : 1.0 for models 1/2; phi^3/(1-phi)^2 for 3
          * phi_old[i], phi_new[i]       : 1.0 (model 1); phi (model 2);
                                           relaxed effective porosity (model 3)
          * EOS scalars                  : ap0/ap1/ap2 (model 1) and
                                           p_a/p_init/c/exp/rho_air/rho_resin (2,3)
        """
        self.validate()
        self._clear_results()
        st = self._settings
        mesh = self._mesh
        resin = self._resin
        stacks, stack_id = self.get_laminate_map()
        i_model = self._i_model
        p_ref, rho_ref, gamma_air = self._air
        thermal_on = self._thermal is not None
        cure_on = self._cure_enabled
        th = self._thermal or {}
        cure = resin.get_cure_kinetics() or {}
        vm = resin.get_viscosity_model()
        N = mesh.N

        neighbours, celltype = create_faces(mesh, st.max_neighbours)
        inlet_cells_all = []
        cascade_events = []
        for port in self._ports:
            if port.kind == "vent":
                celltype[port.cells] = CELL_OUTLET
            elif port.t_activate == 0.0:
                celltype[port.cells] = CELL_INLET
                inlet_cells_all.append(port.cells)
            else:
                cascade_events.append((port.t_activate, port.cells))
                inlet_cells_all.append(port.cells)
        if not (celltype == CELL_INLET).any():
            raise RuntimeError("No inlet cells defined")

        # Bodies without any injection port can never fill: warn and leave
        # them out of the fill criterion.
        body = mesh.get_body_ids()
        fed_bodies = np.unique(body[np.concatenate(inlet_cells_all)])
        fill_region = np.isin(body, fed_bodies)
        if not fill_region.all():
            n_dry = int((~fill_region).sum())
            n_bodies = len(np.setdiff1d(np.unique(body), fed_bodies))
            warnings.warn(
                f"{n_bodies} mesh body(ies) ({n_dry} cells) have no injection "
                f"port and will stay dry; they are excluded from the fill "
                f"criterion.", RuntimeWarning, stacklevel=2)

        # Per-element laminate scalars: each stack's value mapped to its cells.
        def per_element(fn):
            return np.array([fn(s) for s in stacks], dtype=np.float64)[stack_id]

        thickness = per_element(LaminateStack.get_total_thickness)
        porosity = per_element(LaminateStack.get_effective_porosity)
        phi0 = porosity.copy()
        c_porosity = per_element(LaminateStack.get_effective_c_porosity)
        viscosity = (np.full(N, resin.get_viscosity()) if not thermal_on
                     else None)

        geom = create_coordinate_systems(
            mesh, neighbours, celltype, thickness, st.max_neighbours,
        )
        # Drop the unused neighbour columns (padding after the last real
        # neighbour): the kernels stop at the first -9, so results are the
        # same and the per-cell rows fit in fewer cache lines.
        k_used = max(int((geom.neighbours >= 0).sum(axis=1).max()), 1)
        for name in ("neighbours", "cc_to_cc_x", "cc_to_cc_y", "T11", "T12",
                     "T21", "T22", "face_nx", "face_ny", "face_area"):
            setattr(geom, name,
                    np.ascontiguousarray(getattr(geom, name)[:, :k_used]))

        Kxx_base, Kxy_base, Kyy_base = build_stack_tensor(
            mesh, stacks, stack_id, self._fibre_deviation)

        # ---- choose EOS exponent (auto-bump for race-tracking) ----
        K_eig = _eigmax_K(Kxx_base, Kxy_base, Kyy_base)
        K_max = float(np.max(K_eig))
        K_min = float(np.min(K_eig[K_eig > 0])) if np.any(K_eig > 0) else K_max
        perm_ratio = K_max / max(K_min, 1e-30)
        racetrack = perm_ratio >= st.racetrack_perm_ratio
        if st.exp_eos > 0:
            exp_eos = int(st.exp_eos)
        else:
            exp_eos = int(st.exp_eos_racetrack if racetrack
                          else st.exp_eos_default)
        betat2_fac = st.racetrack_dt_factor if racetrack else 1.0

        # ---- pressure normalization & EOS coefficients ----
        rho_resin = resin.get_density()
        rho_resin_param = 0.0 if rho_resin is None else rho_resin
        rho_air_param = rho_ref
        if i_model == 1:
            p_eps = st.p_eps
            p_a = self._p_inlet - self._p_init + p_eps
            p_init_run = p_eps
            ap, rho_a, rho_init = _setup_eos_model1(
                p_ref, rho_ref, gamma_air, st.eos_fit_pressures,
                p_a, p_init_run)
            ap0, ap1, ap2 = float(ap[0]), float(ap[1]), float(ap[2])
            c_eos = 0.0
        else:
            # i_model 2 or 3: use absolute pressures, two-fluid EOS
            p_a = self._p_inlet
            p_init_run = self._p_init
            c_eos, rho_a, rho_init = _setup_eos_model23(
                p_a, p_init_run, rho_air_param, rho_resin, exp_eos)
            ap0 = ap1 = ap2 = 0.0
        p_a_eos = p_a
        p_init_eos = p_init_run

        rho, u, v, p, gamma_vof = _initial_conditions(
            N, celltype, rho_a, rho_init, p_a, p_init_run)

        # ---- per-cell porosity / perm-factor scratch arrays ----
        phi_lo, phi_hi = st.porosity_clip
        if i_model == 1:
            phi_old = np.ones(N)
            phi_new = np.ones(N)
            perm_factor = np.ones(N)
        elif i_model == 2:
            phi_old = porosity.copy()
            phi_new = porosity.copy()
            perm_factor = np.ones(N)
        else:  # i_model == 3
            # Initialize phi_eff_old to the volume-conserving target evaluated
            # at the initial pressure. For all interior cells p == p_init,
            # so phi_p == phi0 + c*p_init^2 (small perturbation).
            phi_p_init = np.clip(phi0 + c_porosity * p_init_run ** 2,
                                 *st.porosity_clip_init)
            phi_target_init = (1.0 - phi0) / (1.0 - phi_p_init) * phi_p_init
            phi_old = phi_target_init.copy()
            phi_new = phi_target_init.copy()
            perm_factor = phi_p_init ** 3 / (1.0 - phi_p_init) ** 2

        # ---- thermal / cure state ----
        if thermal_on:
            alpha_gel = vm["alpha_gel"] if cure_on else 1.0
            C1 = vm["C1"] if cure_on else 0.0
            C2 = vm["C2"] if cure_on else 0.0
            alpha_init = cure.get("alpha_init", 0.0)
            T_field = np.full(N, th["T_init"], dtype=np.float64)
            T_field[celltype == CELL_INLET] = th["T_inlet"]
            alpha_field = np.full(N, alpha_init, dtype=np.float64)
            viscosity = _viscosity_TA(
                T_field, alpha_field, vm["mu_inf"], vm["E_mu"], alpha_gel,
                C1, C2, vm["mu_max"], st.mu_floor, cure_on,
            )
            cp_resin = resin.get_specific_heat()
            rho_fiber = per_element(lambda s: s.get_effective_fibre_property(
                FabricMaterial.get_density))
            cp_fiber = per_element(lambda s: s.get_effective_fibre_property(
                FabricMaterial.get_specific_heat))
        else:
            T_field = np.empty(0)
            alpha_field = np.empty(0)

        # ---- CFL length scale ----
        # h_floor clamps the cell size used in the CFL estimate; 0 keeps
        # the strict minimum ("min" mode).
        cell_area = geom.volume / thickness
        if st.h_min_mode == "percentile":
            h_floor = float(np.sqrt(np.percentile(cell_area,
                                                  st.h_min_percentile)))
        else:
            h_floor = 0.0

        # ---- initial dt (CFL on Darcy-driven max velocity) ----
        area_min = float(np.min(cell_area))
        h_min = max(np.sqrt(area_min), h_floor)
        dp = p_a - p_init_run
        K_eff_max = K_max  # eigmax already reflects in-plane stack tensor
        if i_model == 3:
            K_eff_max *= float(np.max(perm_factor))
        mu_min = float(np.min(viscosity))
        u_max = K_eff_max * dp / (mu_min * h_min) + 1e-30
        dt = st.cfl * betat2_fac * h_min / u_max
        # Thermal stability cap: dt <= cp_eff * h_thk / (2 * h_tool) keeps the
        # implicit Newton-cooling step well-conditioned even though it's
        # unconditionally stable. Convective CFL on T is already covered by
        # the flow CFL since T is advected with u.
        if thermal_on and th["h_tool"] > 0:
            h_thk_min = float(np.min(thickness))
            rcp_min = rho_resin * cp_resin
            dt_therm = (st.dt_thermal_safety * rcp_min * h_thk_min
                        / (2.0 * th["h_tool"] + 1e-30))
            dt = min(dt, dt_therm)

        n_pics = max(4, (self._n_pics // 4) * 4)
        t_max = max(self._tmax, n_pics * dt)
        dt_snap = t_max / n_pics
        # Absolute dt cap: at least min_steps_per_snapshot steps per
        # snapshot interval. Two-fluid models (i_model 2/3) have stiffer
        # dynamics during transient ramp-up, so the cap is the only thing
        # keeping rho from overshooting rho_resin in the first few steps.
        dt_pic_max = t_max / max(st.min_steps_per_snapshot * n_pics, 1)
        dt = min(dt, dt_pic_max)
        dt0 = dt
        # Growth multiplier for the adaptive dt: model 1 is forgiving, the
        # two-fluid models are not.
        dt_growth = st.dt_growth_rtm if i_model == 1 else st.dt_growth_two_fluid

        # Absolute-pressure offset: i_model=1 stores p in normalized form
        # (origin shifted by p_eps), i_model=2/3 store absolute pressure.
        if i_model == 1:
            p_offset = float(self._p_init - p_init_run)
        else:
            p_offset = 0.0

        # Fill-time field: time at which gamma first reaches the threshold.
        g_fill = st.fill_time_threshold
        fill_time = np.full(N, np.nan)
        fill_time[celltype == CELL_INLET] = 0.0

        snapshots = self._snapshots
        def take_snapshot(step, t):
            if i_model == 3:
                phi_p_now = np.clip(phi0 + c_porosity * p ** 2, phi_lo, phi_hi)
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
             for t_a, cids in cascade_events],
            key=lambda e: e[0],
        )

        def _activate_cascade(cids, t):
            celltype[cids] = CELL_INLET
            rho[cids] = rho_a
            p[cids] = p_a
            gamma_vof[cids] = 1.0
            u[cids] = 0.0
            v[cids] = 0.0
            fill_time[cids] = np.where(np.isnan(fill_time[cids]), t,
                                       fill_time[cids])
            if thermal_on:
                T_field[cids] = th["T_inlet"]
                alpha_field[cids] = alpha_init

        # Scalars for the compiled loop (unused ones get neutral values).
        if not thermal_on:
            alpha_gel = C1 = C2 = 0.0
            cp_resin = 0.0
            rho_fiber = cp_fiber = np.zeros(N)
        h_cfl = np.maximum(np.sqrt(geom.volume / thickness), h_floor)
        cfl_fac = st.cfl * betat2_fac

        take_snapshot(0, 0.0)
        t_next = dt_snap
        t = 0.0
        step = 0
        complete = False
        while True:
            t_cascade = pending_cascade[0][0] if pending_cascade else np.inf
            code, t, step, dt, snap_due, fill_done = _advance_jit(
                i_model, thermal_on, cure_on,
                geom.celltype, geom.neighbours, geom.volume,
                geom.cc_to_cc_x, geom.cc_to_cc_y,
                geom.face_nx, geom.face_ny, geom.face_area,
                geom.T11, geom.T12, geom.T21, geom.T22,
                Kxx_base, Kxy_base, Kyy_base, perm_factor, viscosity,
                phi_old, phi_new, phi0, c_porosity,
                float(phi_lo), float(phi_hi), float(st.porosity_relaxation),
                rho, u, v, p, gamma_vof,
                ap0, ap1, ap2,
                float(p_a_eos), float(p_init_eos), float(c_eos), exp_eos,
                float(rho_air_param), float(rho_resin_param),
                thickness, porosity, T_field, alpha_field,
                float(cp_resin), rho_fiber, cp_fiber,
                float(th.get("h_tool", 0.0)), float(th.get("T_tool", 0.0)),
                float(vm.get("mu_inf") or 0.0), float(vm.get("E_mu") or 0.0),
                float(alpha_gel), float(C1), float(C2),
                float(vm.get("mu_max") or 0.0), float(st.mu_floor),
                float(cure.get("H_total", 0.0)),
                float(cure.get("A1", 0.0)), float(cure.get("A2", 0.0)),
                float(cure.get("E1", 0.0)), float(cure.get("E2", 0.0)),
                float(cure.get("m", 0.0)), float(cure.get("n", 0.0)),
                float(st.max_dalpha_per_step),
                fill_time, float(g_fill), fill_region, h_cfl,
                float(t), int(step), float(dt), float(dt0), float(dt_growth),
                float(dt_pic_max), float(cfl_fac), int(st.dt_adapt_after_steps),
                float(t_max), float(t_next), float(t_cascade),
                float(st.fill_stop_fraction),
            )
            if code == ADV_TMAX:
                break
            if code == ADV_CASCADE:
                # Activate any cascade injection ports whose t_activate has passed.
                while pending_cascade and t >= pending_cascade[0][0]:
                    _, cids = pending_cascade.pop(0)
                    _activate_cascade(cids, t)
                continue
            if code == ADV_DIVERGED:
                bad = np.where(~np.isfinite(u) | ~np.isfinite(rho))[0]
                raise FloatingPointError(
                    f"Solver diverged at t={t:.4g} s (step {step}), first "
                    f"non-finite cells: {bad[:10].tolist()}. Check mesh "
                    f"quality (ShellMesh.get_quality_report): needle or "
                    f"strongly non-orthogonal triangles need remeshing.")
            if snap_due:
                take_snapshot(step, t)
                t_next += dt_snap
            if fill_done:
                take_snapshot(step, t)
                complete = True
                break

        self._fill_time = fill_time
        self._fill_region = fill_region
        self._fill_complete = complete
        self._t_end = t
        return snapshots

    # ------------------------------------------------------------------
    # Result getters
    # ------------------------------------------------------------------
    def _require_results(self):
        if not self._snapshots:
            raise RuntimeError("No results: call run() first")

    def get_snapshots(self):
        self._require_results()
        return list(self._snapshots)

    def get_final_snapshot(self):
        self._require_results()
        return self._snapshots[-1]

    def get_snapshot_at(self, t):
        """Snapshot closest in time to t [s]."""
        self._require_results()
        times = np.array([s.t for s in self._snapshots])
        return self._snapshots[int(np.argmin(np.abs(times - t)))]

    def is_fill_complete(self):
        """True when the run stopped on the fill criterion (not on tmax)."""
        self._require_results()
        return self._fill_complete

    def get_total_fill_time(self):
        """
        Time [s] at which the fill criterion was met. If the run ended on
        tmax instead, returns the end time (check is_fill_complete()).
        """
        self._require_results()
        return float(self._t_end)

    def get_fill_time_field(self):
        """Per-cell time [s] at which gamma reached fill_time_threshold (NaN = never)."""
        self._require_results()
        return self._fill_time.copy()

    def get_fill_region(self):
        """Cells that belong to a body fed by at least one injection port."""
        self._require_results()
        return self._fill_region.copy()

    def get_filled_mask_at(self, t):
        """Cells filled (gamma >= threshold) at time t [s], from the fill-time field."""
        self._require_results()
        ft = self._fill_time
        return ~np.isnan(ft) & (ft <= t)

    def get_fill_curve(self):
        """(times, mean fill fraction of the fed fluid cells) per snapshot."""
        self._require_results()
        times = np.array([s.t for s in self._snapshots])
        frac = np.array([s.gamma[s.get_fluid_mask() & self._fill_region].mean()
                         for s in self._snapshots])
        return times, frac

    def get_gamma_at(self, t):
        """Filling fraction field of the snapshot closest to t."""
        return self.get_snapshot_at(t).gamma.copy()

    def get_pressure_at(self, t):
        """Absolute pressure field [Pa] of the snapshot closest to t."""
        return self.get_snapshot_at(t).get_pressure_absolute()

    def get_temperature_at(self, t):
        snap = self.get_snapshot_at(t)
        return None if snap.T is None else snap.T.copy()

    def get_cure_at(self, t):
        snap = self.get_snapshot_at(t)
        return None if snap.alpha is None else snap.alpha.copy()

    def get_viscosity_at(self, t):
        snap = self.get_snapshot_at(t)
        return None if snap.mu is None else snap.mu.copy()

    def get_pressure_results(self):
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
        snaps = self.get_snapshots()
        times = np.array([s.t for s in snaps], dtype=np.float64)
        p_stack = np.array([s.get_pressure_absolute() for s in snaps],
                           dtype=np.float64)
        ct_stack = np.array([s.celltype for s in snaps])
        p_min = np.empty(len(snaps))
        p_max = np.empty(len(snaps))
        p_mean = np.empty(len(snaps))
        for k, s in enumerate(snaps):
            fluid = s.get_fluid_mask()
            if not fluid.any():
                p_min[k] = p_max[k] = p_mean[k] = np.nan
                continue
            pa = s.get_pressure_absolute()[fluid]
            p_min[k] = pa.min()
            p_max[k] = pa.max()
            p_mean[k] = pa.mean()
        return dict(times=times, p=p_stack, celltype=ct_stack,
                    p_min=p_min, p_max=p_max, p_mean=p_mean)
