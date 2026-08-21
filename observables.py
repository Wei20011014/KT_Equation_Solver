"""Physical observables for the six D1 -> D pi pi charge channels.

This module is the layer above ``amplitude.py``.  It converts the physical
helicity amplitudes into

* the unpolarized squared matrix element,
* Dalitz-plot densities d^2 Gamma / (ds dt) and d^2 Gamma / (ds dz_s),
* the dipion invariant-mass spectra d Gamma / ds and d Gamma / dm_pipi,
* integrated partial widths and branching fractions.

Conventions
-----------
The helicity amplitudes returned by ``amplitude.py`` contain the full decay
amplitude in the convention of ``D1Dpipi.pdf``.  For an unpolarized spin-one
initial state,

    |M|^2_bar = (|H0|^2 + |H+|^2 + |H-|^2) / 3
              = (|H0|^2 + |Hperp|^2) / 3.

The standard three-body phase-space normalization is

    d^2 Gamma / (ds dt)
        = S_id |M|^2_bar / (256 pi^3 m_D1^3),

where S_id = 1/2! for the two pi0-pi0 channels and S_id = 1 otherwise.
The factor 1/2! assumes integration over the complete labelled Dalitz region;
do not include it again in external integration code.

All masses and widths are in GeV, while s, t and u are in GeV^2.  The current
implementation uses the isospin-averaged masses in ``kinematics.py`` for all
six charge channels and is restricted to the real decay region.
"""

from functools import lru_cache
from typing import Mapping

import numpy as np

import amplitude as amp
import kinematics as kin


THREE_BODY_DALITZ_PREFACTOR = 1.0 / (
    256.0 * np.pi**3 * kin.m_D1**3
)


# ---------------------------------------------------------------------------
# Validation and small utilities
# ---------------------------------------------------------------------------

def _validate_process(process):
    if process not in amp.PROCESS_INFO:
        raise ValueError(
            f"Unknown process {process!r}. Choose one of {amp.PROCESS_ORDER}."
        )


def _finite_real_array(x, name):
    value = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must contain finite real values.")
    return value


def _finite_complex_array(x, name):
    value = np.asarray(x, dtype=complex)
    if np.any(~np.isfinite(value.real)) or np.any(~np.isfinite(value.imag)):
        raise ValueError(f"{name} must contain finite values.")
    return value


def _abs2(x):
    x = np.asarray(x, dtype=complex)
    return x.real**2 + x.imag**2


def _scalar_if_scalar(reference, value):
    value = np.asarray(value)
    if np.asarray(reference).ndim == 0:
        return value.item()
    return value


def _validate_s_region(s, allow_endpoints=True, tolerance=2.0e-12):
    s = _finite_real_array(s, "s")
    scale = max(1.0, abs(kin.s_decay_max))
    lower = kin.s_decay_min
    upper = kin.s_decay_max

    if np.any(s < lower - tolerance * scale) or np.any(
        s > upper + tolerance * scale
    ):
        raise ValueError(
            "s lies outside the physical D1 -> D pi pi decay interval "
            f"[{lower}, {upper}] GeV^2."
        )

    if not allow_endpoints and np.any((s <= lower) | (s >= upper)):
        raise ValueError(
            "This evaluation requires an interior s value.  The exact decay "
            "endpoints contain vanishing kinematic factors."
        )

    return np.clip(s, lower, upper)


def _check_solution_domain(solution, s, t, u, tolerance=1.0e-12):
    """Give an informative error before a non-extrapolating interpolator does."""
    if not isinstance(solution, Mapping) or "grids" not in solution:
        return

    grids = solution["grids"]
    if not isinstance(grids, Mapping):
        return

    variables = {"M": s, "N": t, "R": u}
    for family, variable in variables.items():
        if family not in grids:
            continue
        grid = np.asarray(grids[family], dtype=float)
        if grid.ndim != 1 or grid.size == 0:
            continue

        scale = max(1.0, abs(grid[0]), abs(grid[-1]))
        if np.any(variable < grid[0] - tolerance * scale) or np.any(
            variable > grid[-1] + tolerance * scale
        ):
            raise ValueError(
                f"The requested Dalitz point reaches outside solution['grids']"
                f"[{family!r}] = [{grid[0]}, {grid[-1]}].  Build the KT grid "
                "closer to the physical endpoints, enable state extrapolation "
                "when solving, or use a less endpoint-focused integration grid."
            )


@lru_cache(maxsize=None)
def _legendre_grid(n_points):
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("The quadrature order must be an integer >= 2.")
    return np.polynomial.legendre.leggauss(int(n_points))


def _mapped_legendre_grid(lower, upper, n_points):
    nodes, weights = _legendre_grid(int(n_points))
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return midpoint + half_width * nodes, half_width * weights


# ---------------------------------------------------------------------------
# Spin sums and symmetry factors
# ---------------------------------------------------------------------------

def identical_particle_factor(process):
    """Return 1/2! for pi0-pi0 and 1 for the other four processes."""
    _validate_process(process)
    return 0.5 if amp.PROCESS_INFO[process]["identical_pions"] else 1.0


def polarization_summed_matrix_element_squared(helicity):
    """Return |H0|^2 + |H+|^2 + |H-|^2.

    A mapping containing ``H0`` and either both ``Hplus``, ``Hminus`` or the
    independent transverse combination ``Hperp`` is accepted.
    """
    if not isinstance(helicity, Mapping) or "H0" not in helicity:
        raise ValueError("helicity must be a mapping containing H0.")

    H0 = _finite_complex_array(helicity["H0"], "H0")
    has_plus = "Hplus" in helicity
    has_minus = "Hminus" in helicity

    if has_plus != has_minus:
        raise ValueError("Supply both Hplus and Hminus, or supply Hperp.")

    if has_plus:
        Hplus = _finite_complex_array(helicity["Hplus"], "Hplus")
        Hminus = _finite_complex_array(helicity["Hminus"], "Hminus")
        H0, Hplus, Hminus = np.broadcast_arrays(H0, Hplus, Hminus)
        return _abs2(H0) + _abs2(Hplus) + _abs2(Hminus)

    if "Hperp" not in helicity:
        raise ValueError("Supply Hplus and Hminus, or supply Hperp.")

    Hperp = _finite_complex_array(helicity["Hperp"], "Hperp")
    H0, Hperp = np.broadcast_arrays(H0, Hperp)
    return _abs2(H0) + _abs2(Hperp)


def unpolarized_matrix_element_squared_from_helicity(helicity):
    """Return the spin-averaged squared amplitude for an unpolarized D1."""
    return polarization_summed_matrix_element_squared(helicity) / 3.0


def unpolarized_matrix_element_squared(s, t, u, solution, process):
    """Reconstruct one physical channel and return its unpolarized |M|^2."""
    _validate_process(process)
    s, t, u = amp.prepare_invariants(s, t, u)
    _check_solution_domain(solution, s, t, u)
    helicity = amp.physical_helicity_amplitudes(
        s,
        t,
        u,
        solution,
        process,
    )
    return unpolarized_matrix_element_squared_from_helicity(helicity)


def all_unpolarized_matrix_elements_squared(s, t, u, solution):
    """Return unpolarized |M|^2 for all six physical charge processes."""
    s, t, u = amp.prepare_invariants(s, t, u)
    _check_solution_domain(solution, s, t, u)
    helicities = amp.all_physical_helicity_amplitudes(s, t, u, solution)
    return {
        process: unpolarized_matrix_element_squared_from_helicity(values)
        for process, values in helicities.items()
    }


# ---------------------------------------------------------------------------
# Differential decay rates
# ---------------------------------------------------------------------------

def dt_dz_s_absolute(s):
    """Return |dt/dz_s| at fixed s in GeV^2."""
    s = _validate_s_region(s, allow_endpoints=True)
    lambda_parent = kin.physical_sqrt(
        kin.kallen(s, kin.m_D1**2, kin.m_D**2)
    )
    lambda_pions = kin.physical_sqrt(
        kin.kallen(s, kin.m_pi**2, kin.m_pi**2)
    )
    result = lambda_parent * lambda_pions / (2.0 * s)
    return _scalar_if_scalar(s, result)


def differential_width_dsdt(s, t, solution, process, u=None):
    """Return d^2 Gamma/(ds dt) for one charge channel in GeV^(-3).

    ``u`` may be omitted, in which case it is calculated from
    ``u = Sigma - s - t``.  The returned rate already contains the 1/2!
    factor for a pi0-pi0 final state.
    """
    _validate_process(process)
    s, t, u = amp.prepare_invariants(s, t, u)
    _check_solution_domain(solution, s, t, u)
    matrix_element_squared = unpolarized_matrix_element_squared(
        s,
        t,
        u,
        solution,
        process,
    )
    return (
        identical_particle_factor(process)
        * THREE_BODY_DALITZ_PREFACTOR
        * matrix_element_squared
    )


def all_differential_widths_dsdt(s, t, solution, u=None):
    """Return d^2 Gamma/(ds dt) for all six charge channels."""
    s, t, u = amp.prepare_invariants(s, t, u)
    _check_solution_domain(solution, s, t, u)
    squared = all_unpolarized_matrix_elements_squared(s, t, u, solution)
    return {
        process: (
            identical_particle_factor(process)
            * THREE_BODY_DALITZ_PREFACTOR
            * value
        )
        for process, value in squared.items()
    }


def differential_width_dsdz(s, z_s, solution, process):
    """Return d^2 Gamma/(ds dz_s) for -1 <= z_s <= 1 in GeV^(-1)."""
    _validate_process(process)
    s = _validate_s_region(s, allow_endpoints=False)
    z_s = _finite_real_array(z_s, "z_s")
    s, z_s = np.broadcast_arrays(s, z_s)

    tolerance = 2.0e-12
    if np.any(np.abs(z_s) > 1.0 + tolerance):
        raise ValueError("z_s must lie in [-1, 1].")
    z_s = np.clip(z_s, -1.0, 1.0)

    t = kin.t_of_s_z(s, z_s)
    u = kin.u_of_s_z(s, z_s)
    return differential_width_dsdt(
        s,
        t,
        solution,
        process,
        u=u,
    ) * dt_dz_s_absolute(s)


def all_differential_widths_dsdz(s, z_s, solution):
    """Return d^2 Gamma/(ds dz_s) for all six charge channels."""
    s = _validate_s_region(s, allow_endpoints=False)
    z_s = _finite_real_array(z_s, "z_s")
    s, z_s = np.broadcast_arrays(s, z_s)

    tolerance = 2.0e-12
    if np.any(np.abs(z_s) > 1.0 + tolerance):
        raise ValueError("z_s must lie in [-1, 1].")
    z_s = np.clip(z_s, -1.0, 1.0)

    t = kin.t_of_s_z(s, z_s)
    u = kin.u_of_s_z(s, z_s)
    jacobian = dt_dz_s_absolute(s)
    dsdt = all_differential_widths_dsdt(
        s,
        t,
        solution,
        u=u,
    )
    return {
        process: value * jacobian
        for process, value in dsdt.items()
    }


# ---------------------------------------------------------------------------
# One-dimensional spectra and integrated widths
# ---------------------------------------------------------------------------

def differential_width_ds(s, solution, process, n_z=48):
    """Integrate over z_s and return d Gamma/ds in GeV^(-1).

    Exact s endpoints are returned as zero.  This avoids evaluating the
    kinematically reduced amplitudes at their singular basis endpoints.
    """
    _validate_process(process)
    original_s = _finite_real_array(s, "s")
    s_array = _validate_s_region(original_s, allow_endpoints=True)
    flat_s = s_array.reshape(-1)
    result = np.zeros(flat_s.shape, dtype=float)

    interior = (
        (flat_s > kin.s_decay_min)
        & (flat_s < kin.s_decay_max)
    )
    if np.any(interior):
        z_nodes, z_weights = _legendre_grid(int(n_z))
        s_mesh = flat_s[interior, None]
        z_mesh = z_nodes[None, :]
        density = differential_width_dsdz(
            s_mesh,
            z_mesh,
            solution,
            process,
        )
        result[interior] = np.sum(density * z_weights[None, :], axis=-1)

    result = result.reshape(s_array.shape)
    return _scalar_if_scalar(original_s, result)


def all_differential_widths_ds(s, solution, n_z=48):
    """Return d Gamma/ds for all six charge channels."""
    return {
        process: differential_width_ds(
            s,
            solution,
            process,
            n_z=n_z,
        )
        for process in amp.PROCESS_ORDER
    }


def dipion_mass_spectrum(m_pipi, solution, process, n_z=48):
    """Return d Gamma/dm_pipi in GeV^0, with m_pipi supplied in GeV."""
    mass = _finite_real_array(m_pipi, "m_pipi")
    if np.any(mass < 0.0):
        raise ValueError("m_pipi must be non-negative.")
    result = 2.0 * mass * differential_width_ds(
        mass**2,
        solution,
        process,
        n_z=n_z,
    )
    return _scalar_if_scalar(m_pipi, result)


def all_dipion_mass_spectra(m_pipi, solution, n_z=48):
    """Return d Gamma/dm_pipi for all six charge channels."""
    return {
        process: dipion_mass_spectrum(
            m_pipi,
            solution,
            process,
            n_z=n_z,
        )
        for process in amp.PROCESS_ORDER
    }


def total_width(solution, process, n_s=48, n_z=48):
    """Return the integrated partial width Gamma in GeV.

    Gauss-Legendre nodes do not touch the kinematic endpoints.  If a KT
    solution was built on a noticeably inset grid and disallows
    extrapolation, a domain error explains how to enlarge the usable grid.
    """
    _validate_process(process)
    s_nodes, s_weights = _mapped_legendre_grid(
        kin.s_decay_min,
        kin.s_decay_max,
        int(n_s),
    )
    z_nodes, z_weights = _legendre_grid(int(n_z))

    density = differential_width_dsdz(
        s_nodes[:, None],
        z_nodes[None, :],
        solution,
        process,
    )
    value = np.sum(
        s_weights[:, None] * z_weights[None, :] * density
    )
    return float(np.real(value))


def all_total_widths(solution, n_s=48, n_z=48):
    """Return integrated partial widths in GeV for all six processes."""
    return {
        process: total_width(
            solution,
            process,
            n_s=n_s,
            n_z=n_z,
        )
        for process in amp.PROCESS_ORDER
    }


def branching_fraction(partial_width, parent_total_width):
    """Return Gamma_partial/Gamma_total for widths in the same units."""
    partial_width = _finite_real_array(partial_width, "partial_width")
    parent_total_width = _finite_real_array(
        parent_total_width,
        "parent_total_width",
    )
    partial_width, parent_total_width = np.broadcast_arrays(
        partial_width,
        parent_total_width,
    )
    if np.any(partial_width < 0.0):
        raise ValueError("partial_width must be non-negative.")
    if np.any(parent_total_width <= 0.0):
        raise ValueError("parent_total_width must be positive.")
    result = partial_width / parent_total_width
    return _scalar_if_scalar(partial_width, result)


# ---------------------------------------------------------------------------
# Plot-ready Dalitz data
# ---------------------------------------------------------------------------

def dalitz_plot_data(
    solution,
    process,
    n_s=120,
    n_z=120,
    edge_fraction=5.0e-4,
):
    """Return a rectangular (s,z_s) mesh mapped into the physical Dalitz plot.

    The dictionary contains ``s``, ``z_s``, ``t``, ``u``, ``dGamma_dsdt`` and
    ``dGamma_dsdz`` arrays.  A small endpoint offset avoids the singular basis
    factors in the current real-axis amplitude reconstruction.
    """
    _validate_process(process)
    if not isinstance(n_s, (int, np.integer)) or n_s < 2:
        raise ValueError("n_s must be an integer >= 2.")
    if not isinstance(n_z, (int, np.integer)) or n_z < 2:
        raise ValueError("n_z must be an integer >= 2.")
    if not 0.0 < edge_fraction < 0.1:
        raise ValueError("edge_fraction must lie between 0 and 0.1.")

    s_width = kin.s_decay_max - kin.s_decay_min
    s_values = np.linspace(
        kin.s_decay_min + edge_fraction * s_width,
        kin.s_decay_max - edge_fraction * s_width,
        int(n_s),
    )
    z_values = np.linspace(
        -1.0 + edge_fraction,
        1.0 - edge_fraction,
        int(n_z),
    )
    s_mesh, z_mesh = np.meshgrid(s_values, z_values, indexing="ij")
    t_mesh = kin.t_of_s_z(s_mesh, z_mesh)
    u_mesh = kin.u_of_s_z(s_mesh, z_mesh)
    dsdt = differential_width_dsdt(
        s_mesh,
        t_mesh,
        solution,
        process,
        u=u_mesh,
    )

    return {
        "s": s_mesh,
        "z_s": z_mesh,
        "t": t_mesh,
        "u": u_mesh,
        "dGamma_dsdt": dsdt,
        "dGamma_dsdz": dsdt * dt_dz_s_absolute(s_mesh),
    }


# ---------------------------------------------------------------------------
# Internal consistency checks
# ---------------------------------------------------------------------------

def run_self_checks(atol=2.0e-11):
    """Check the spin sum, Jacobian, symmetry factors and positivity."""
    errors = {}

    helicity = {
        "H0": 1.2 - 0.4j,
        "Hplus": -0.3 + 0.2j,
        "Hminus": 0.3 - 0.2j,
        "Hperp": np.sqrt(2.0) * (-0.3 + 0.2j),
    }
    expected_sum = (
        abs(helicity["H0"])**2
        + abs(helicity["Hplus"])**2
        + abs(helicity["Hminus"])**2
    )
    calculated_sum = polarization_summed_matrix_element_squared(helicity)
    errors["helicity_spin_sum"] = abs(calculated_sum - expected_sum)

    transverse_only = {
        "H0": helicity["H0"],
        "Hperp": helicity["Hperp"],
    }
    errors["Hperp_equivalence"] = abs(
        polarization_summed_matrix_element_squared(transverse_only)
        - expected_sum
    )
    errors["pi0_symmetry_factor"] = abs(
        identical_particle_factor("D10_to_D0_pi0_pi0") - 0.5
    )
    errors["charged_symmetry_factor"] = abs(
        identical_particle_factor("D10_to_D0_pip_pim") - 1.0
    )

    constant = lambda value: (
        lambda x: value + 0.0j * np.asarray(x)
    )
    toy_solution = {
        "M": {
            "0": {"S0": constant(0.8 - 0.1j)},
            "1": {
                "P0": constant(0.25 + 0.05j),
                "Pperp": constant(-0.12 + 0.02j),
            },
        },
        "N": {},
        "R": {},
    }

    s = 0.5 * (kin.s_decay_min + kin.s_decay_max)
    z_s = 0.27
    t = kin.t_of_s_z(s, z_s)
    u = kin.u_of_s_z(s, z_s)
    process = "D10_to_D0_pip_pim"

    density_st = differential_width_dsdt(
        s,
        t,
        toy_solution,
        process,
        u=u,
    )
    density_sz = differential_width_dsdz(
        s,
        z_s,
        toy_solution,
        process,
    )
    errors["Dalitz_Jacobian"] = abs(
        density_sz - density_st * dt_dz_s_absolute(s)
    )

    width = total_width(toy_solution, process, n_s=10, n_z=10)
    errors["positive_width"] = 0.0 if np.isfinite(width) and width >= 0.0 else 1.0

    maximum = max(float(value) for value in errors.values())
    if maximum > atol:
        raise RuntimeError(f"observables self-check failed: {errors}")
    return errors


__all__ = [
    "THREE_BODY_DALITZ_PREFACTOR",
    "identical_particle_factor",
    "polarization_summed_matrix_element_squared",
    "unpolarized_matrix_element_squared_from_helicity",
    "unpolarized_matrix_element_squared",
    "all_unpolarized_matrix_elements_squared",
    "dt_dz_s_absolute",
    "differential_width_dsdt",
    "all_differential_widths_dsdt",
    "differential_width_dsdz",
    "all_differential_widths_dsdz",
    "differential_width_ds",
    "all_differential_widths_ds",
    "dipion_mass_spectrum",
    "all_dipion_mass_spectra",
    "total_width",
    "all_total_widths",
    "branching_fraction",
    "dalitz_plot_data",
    "run_self_checks",
]
