"""Generic iterative Khuri-Treiman solver for D1 -> D pi pi.

The solver connects

* ``angular_averages.py`` for the crossed-channel hat functions, and
* ``kt_equations.py`` for a one-channel dispersive update.

The retained S/P channel layout is

    M["0"]["S0"]
    M["1"]["P0"], M["1"]["Pperp"]

    N["1/2" or "3/2"]["S0", "P0", "Pperp"]
    R["1/2" or "3/2"]["S0", "P0", "Pperp"]

where M, N and R carry the s-, t- and u-channel right-hand cuts.

This is a real-axis prototype.  It inherits the real-square-root restriction
of the current ``kinematics.py`` and ``angular_averages.py``.  A full solution
outside the real decay region requires a specified analytic continuation or
contour deformation.
"""

from typing import Mapping

import numpy as np

import angular_averages as angular
import kt_equations as kt


# ---------------------------------------------------------------------------
# Retained S/P channels
# ---------------------------------------------------------------------------

ALLOWED_CHANNELS = {
    "M": {
        "0": ("S0",),
        "1": ("P0", "Pperp"),
    },
    "N": {
        "1/2": ("S0", "P0", "Pperp"),
        "3/2": ("S0", "P0", "Pperp"),
    },
    "R": {
        "1/2": ("S0", "P0", "Pperp"),
        "3/2": ("S0", "P0", "Pperp"),
    },
}


# ---------------------------------------------------------------------------
# Public configuration helpers
# ---------------------------------------------------------------------------

def channel_config(
    phase_shift,
    omega,
    threshold,
    cutoff,
    subtraction_coefficients,
    n_subtractions=None,
    subtraction_point=0.0,
    polynomial_center=0.0,
    tail_correction=0.0,
    boundary="upper",
    epsabs=1.0e-8,
    epsrel=1.0e-7,
    integration_limit=200,
    omega_floor=1.0e-14,
):
    """Create and validate one channel's KT configuration dictionary."""
    if not callable(phase_shift):
        raise TypeError("phase_shift must be callable.")
    if not callable(omega):
        raise TypeError("omega must be callable.")

    threshold = float(threshold)
    cutoff = float(cutoff)
    if not np.isfinite(threshold) or not np.isfinite(cutoff):
        raise ValueError("threshold and cutoff must be finite.")
    if cutoff <= threshold:
        raise ValueError("cutoff must be greater than threshold.")

    coefficients = np.asarray(subtraction_coefficients, dtype=complex)
    if coefficients.ndim != 1:
        raise ValueError("subtraction_coefficients must be one-dimensional.")
    if np.any(~np.isfinite(coefficients.real)) or np.any(
        ~np.isfinite(coefficients.imag)
    ):
        raise ValueError("subtraction_coefficients must be finite.")

    if n_subtractions is None:
        n_subtractions = int(coefficients.size)
    if not isinstance(n_subtractions, (int, np.integer)):
        raise TypeError("n_subtractions must be an integer.")
    if n_subtractions < 0:
        raise ValueError("n_subtractions must be non-negative.")
    if coefficients.size > n_subtractions:
        raise ValueError(
            "n_subtractions must be at least the number of polynomial "
            "coefficients."
        )

    subtraction_point = float(subtraction_point)
    if n_subtractions > 0 and threshold <= subtraction_point <= cutoff:
        raise ValueError(
            "subtraction_point must lie outside the integration interval."
        )
    if boundary not in {"upper", "lower", "principal_value"}:
        raise ValueError(
            "boundary must be 'upper', 'lower', or 'principal_value'."
        )
    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("epsabs and epsrel must be positive.")
    if not isinstance(integration_limit, (int, np.integer)) or integration_limit < 1:
        raise ValueError("integration_limit must be a positive integer.")
    if omega_floor <= 0.0:
        raise ValueError("omega_floor must be positive.")

    return {
        "phase_shift": phase_shift,
        "omega": omega,
        "threshold": threshold,
        "cutoff": cutoff,
        "subtraction_coefficients": coefficients.copy(),
        "n_subtractions": int(n_subtractions),
        "subtraction_point": subtraction_point,
        "polynomial_center": float(polynomial_center),
        "tail_correction": tail_correction,
        "boundary": boundary,
        "epsabs": float(epsabs),
        "epsrel": float(epsrel),
        "integration_limit": int(integration_limit),
        "omega_floor": float(omega_floor),
    }


def make_solver_grid(
    physical_minimum,
    physical_maximum,
    n_points,
    edge_fraction=1.0e-4,
):
    """Create a real-decay grid and safe effective integration limits.

    The returned grid extends slightly beyond the returned integration
    interval.  This lets the hat-function interpolator cover every quadrature
    point while avoiding evaluations exactly at a kinematic endpoint.

    Returns
    -------
    grid, effective_threshold, effective_cutoff
    """
    physical_minimum = float(physical_minimum)
    physical_maximum = float(physical_maximum)

    if physical_maximum <= physical_minimum:
        raise ValueError("physical_maximum must exceed physical_minimum.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 4:
        raise ValueError("n_points must be an integer greater than or equal to 4.")
    if not 0.0 < edge_fraction < 0.1:
        raise ValueError("edge_fraction must lie between 0 and 0.1.")

    width = physical_maximum - physical_minimum
    edge = edge_fraction * width

    grid = np.linspace(
        physical_minimum + edge,
        physical_maximum - edge,
        int(n_points),
    )
    effective_threshold = physical_minimum + 2.0 * edge
    effective_cutoff = physical_maximum - 2.0 * edge

    return grid, effective_threshold, effective_cutoff


# ---------------------------------------------------------------------------
# Validation and nested-dictionary utilities
# ---------------------------------------------------------------------------

def _iter_channels(channel_configs):
    for family, isospins in channel_configs.items():
        for isospin, waves in isospins.items():
            for wave, config in waves.items():
                yield family, isospin, wave, config


def _validate_channel_layout(channel_configs):
    if not isinstance(channel_configs, Mapping) or not channel_configs:
        raise ValueError("channel_configs must be a non-empty mapping.")

    for family, isospins in channel_configs.items():
        if family not in ALLOWED_CHANNELS:
            raise ValueError(f"Unknown amplitude family {family!r}.")
        if not isinstance(isospins, Mapping):
            raise TypeError(f"channel_configs[{family!r}] must be a mapping.")

        for isospin, waves in isospins.items():
            if isospin not in ALLOWED_CHANNELS[family]:
                raise ValueError(
                    f"Isospin {isospin!r} is not allowed in family {family!r}."
                )
            if not isinstance(waves, Mapping):
                raise TypeError(
                    f"channel_configs[{family!r}][{isospin!r}] must be a mapping."
                )

            for wave, config in waves.items():
                if wave not in ALLOWED_CHANNELS[family][isospin]:
                    raise ValueError(
                        f"Wave {wave!r} is not retained for "
                        f"{family}[{isospin!r}]."
                    )
                if not isinstance(config, Mapping):
                    raise TypeError("Every channel configuration must be a mapping.")

                required = {
                    "phase_shift",
                    "omega",
                    "threshold",
                    "cutoff",
                    "subtraction_coefficients",
                    "n_subtractions",
                    "subtraction_point",
                    "polynomial_center",
                    "tail_correction",
                    "boundary",
                    "epsabs",
                    "epsrel",
                    "integration_limit",
                    "omega_floor",
                }
                missing = required.difference(config)
                if missing:
                    raise ValueError(
                        f"Configuration for {family}/{isospin}/{wave} is missing "
                        f"keys {sorted(missing)}.  Build it with channel_config()."
                    )


def _validate_grids(grids, channel_configs, extrapolate_hats):
    validated = {}

    for family in channel_configs:
        if family not in grids:
            raise ValueError(f"A grid is required for amplitude family {family!r}.")

        grid = np.asarray(grids[family], dtype=float)
        if grid.ndim != 1 or grid.size < 4:
            raise ValueError(f"grids[{family!r}] must be a 1D array with >=4 points.")
        if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"grids[{family!r}] must be finite and increasing.")

        validated[family] = grid.copy()

    for family, _, _, config in _iter_channels(channel_configs):
        grid = validated[family]
        threshold = config["threshold"]
        cutoff = config["cutoff"]
        endpoint_tolerance = 1.0e-13 * max(1.0, abs(threshold), abs(cutoff))

        if np.any(np.abs(grid - threshold) <= endpoint_tolerance) or np.any(
            np.abs(grid - cutoff) <= endpoint_tolerance
        ):
            raise ValueError(
                f"grids[{family!r}] contains an integration endpoint. "
                "Use make_solver_grid() or shift the endpoint by epsilon."
            )

        if not extrapolate_hats and (
            grid[0] > threshold or grid[-1] < cutoff
        ):
            raise ValueError(
                f"grids[{family!r}] must cover [{threshold}, {cutoff}] when "
                "extrapolate_hats=False."
            )

    return validated


def _scalar_complex(entry, x, name):
    value = entry(float(x)) if callable(entry) else entry
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} must return one value for scalar input.")
    result = complex(value.reshape(-1)[0])
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} returned a non-finite value at x={x}.")
    return result


def _sample_initial_entry(entry, grid, name):
    if callable(entry):
        values = np.asarray(entry(grid), dtype=complex)
    else:
        values = np.asarray(entry, dtype=complex)

    try:
        values = np.broadcast_to(values, grid.shape).astype(complex, copy=True)
    except ValueError as error:
        raise ValueError(
            f"Initial entry {name} cannot be broadcast to grid shape {grid.shape}."
        ) from error

    if np.any(~np.isfinite(values.real)) or np.any(~np.isfinite(values.imag)):
        raise ValueError(f"Initial entry {name} contains non-finite values.")
    return values


def _maximum_state_norm(state):
    maximum = 0.0
    for _, isospins in state.items():
        for _, waves in isospins.items():
            for _, values in waves.items():
                maximum = max(maximum, float(np.max(np.abs(values))))
    return maximum


def _maximum_state_difference(left, right):
    maximum = 0.0
    for family, isospins in left.items():
        for isospin, waves in isospins.items():
            for wave, values in waves.items():
                difference = np.max(
                    np.abs(values - right[family][isospin][wave])
                )
                maximum = max(maximum, float(difference))
    return maximum


# ---------------------------------------------------------------------------
# Initial state and interpolation
# ---------------------------------------------------------------------------

def build_initial_state(grids, channel_configs, initial_state=None):
    """Build F^(0)=Omega*(P+tail), with optional user overrides."""
    result = {}

    for family, isospin, wave, config in _iter_channels(channel_configs):
        grid = grids[family]
        override = None
        if initial_state is not None:
            override = (
                initial_state.get(family, {})
                .get(isospin, {})
                .get(wave, None)
            )

        if override is not None:
            values = _sample_initial_entry(
                override,
                grid,
                f"{family}/{isospin}/{wave}",
            )
        else:
            polynomial = kt.subtraction_polynomial(
                grid,
                config["subtraction_coefficients"],
                center=config["polynomial_center"],
            )
            omega_values = np.empty(grid.shape, dtype=complex)
            tail_values = np.empty(grid.shape, dtype=complex)

            for index, x in enumerate(grid):
                omega_values[index] = _scalar_complex(
                    config["omega"],
                    x,
                    "omega",
                )
                tail_values[index] = _scalar_complex(
                    config["tail_correction"],
                    x,
                    "tail_correction",
                )

            values = omega_values * (polynomial + tail_values)

        result.setdefault(family, {}).setdefault(isospin, {})[wave] = values

    return result


def build_state_interpolators(
    grids,
    state,
    kind="cubic",
    extrapolate=False,
):
    """Replace every state array by a callable interpolator."""
    return {
        family: kt.interpolate_nested_hat_functions(
            grids[family],
            isospins,
            kind=kind,
            extrapolate=extrapolate,
        )
        for family, isospins in state.items()
    }


# ---------------------------------------------------------------------------
# Hat functions and one simultaneous iteration
# ---------------------------------------------------------------------------

def compute_hat_values(grids, channel_configs, state_interpolators, n_points=80):
    """Compute every requested M-hat, N-hat and R-hat array."""
    M = state_interpolators.get("M", {})
    N = state_interpolators.get("N", {})
    R = state_interpolators.get("R", {})

    hats = {}
    if "M" in channel_configs:
        hats["M"] = angular.hat_s(
            grids["M"],
            N=N,
            R=R,
            n_points=n_points,
        )
    if "N" in channel_configs:
        hats["N"] = angular.hat_t(
            grids["N"],
            M=M,
            R=R,
            n_points=n_points,
        )
    if "R" in channel_configs:
        hats["R"] = angular.hat_u(
            grids["R"],
            M=M,
            N=N,
            n_points=n_points,
        )

    return hats


def _raw_updated_state(
    grids,
    channel_configs,
    hat_interpolators,
):
    result = {}

    for family, isospin, wave, config in _iter_channels(channel_configs):
        grid = grids[family]
        hat_function = hat_interpolators[family][isospin][wave]

        values = kt.kt_update(
            x=grid,
            phase_shift=config["phase_shift"],
            omega=config["omega"],
            hat_function=hat_function,
            threshold=config["threshold"],
            cutoff=config["cutoff"],
            subtraction_coefficients=config["subtraction_coefficients"],
            n_subtractions=config["n_subtractions"],
            subtraction_point=config["subtraction_point"],
            polynomial_center=config["polynomial_center"],
            tail_correction=config["tail_correction"],
            boundary=config["boundary"],
            epsabs=config["epsabs"],
            epsrel=config["epsrel"],
            integration_limit=config["integration_limit"],
            omega_floor=config["omega_floor"],
        )
        result.setdefault(family, {}).setdefault(isospin, {})[wave] = values

    return result


def kt_iteration(
    grids,
    channel_configs,
    state,
    mixing=0.5,
    angular_points=80,
    interpolation_kind="cubic",
    extrapolate_state=False,
    extrapolate_hats=False,
):
    """Perform one simultaneous M/N/R fixed-point iteration."""
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must satisfy 0 < mixing <= 1.")

    state_interpolators = build_state_interpolators(
        grids,
        state,
        kind=interpolation_kind,
        extrapolate=extrapolate_state,
    )
    hats = compute_hat_values(
        grids,
        channel_configs,
        state_interpolators,
        n_points=angular_points,
    )
    hat_interpolators = {
        family: kt.interpolate_nested_hat_functions(
            grids[family],
            values,
            kind=interpolation_kind,
            extrapolate=extrapolate_hats,
        )
        for family, values in hats.items()
    }
    raw_state = _raw_updated_state(
        grids,
        channel_configs,
        hat_interpolators,
    )

    mixed_state = {}
    for family, isospins in raw_state.items():
        for isospin, waves in isospins.items():
            for wave, raw_values in waves.items():
                old_values = state[family][isospin][wave]
                mixed_values = (
                    (1.0 - mixing) * old_values
                    + mixing * raw_values
                )
                if np.any(~np.isfinite(mixed_values.real)) or np.any(
                    ~np.isfinite(mixed_values.imag)
                ):
                    raise FloatingPointError(
                        f"Non-finite iterate in {family}/{isospin}/{wave}."
                    )
                mixed_state.setdefault(family, {}).setdefault(isospin, {})[
                    wave
                ] = mixed_values

    residual_absolute = _maximum_state_difference(raw_state, state)
    step_absolute = _maximum_state_difference(mixed_state, state)
    scale = max(1.0, _maximum_state_norm(raw_state), _maximum_state_norm(state))

    diagnostics = {
        "residual_absolute": residual_absolute,
        "residual_relative": residual_absolute / scale,
        "step_absolute": step_absolute,
        "step_relative": step_absolute / scale,
        "scale": scale,
    }

    return mixed_state, hats, diagnostics


# ---------------------------------------------------------------------------
# Full fixed-point solver
# ---------------------------------------------------------------------------

def solve_kt(
    grids,
    channel_configs,
    initial_state=None,
    max_iterations=20,
    relative_tolerance=1.0e-6,
    absolute_tolerance=1.0e-9,
    mixing=0.5,
    angular_points=80,
    interpolation_kind="cubic",
    extrapolate_state=False,
    extrapolate_hats=False,
    verbose=True,
):
    """Iterate all configured S/P channels to a fixed point.

    Convergence is tested with the unmixed fixed-point residual, preventing a
    very small mixing parameter from creating a false convergence signal.
    """
    _validate_channel_layout(channel_configs)

    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("Convergence tolerances must be positive.")
    if not isinstance(angular_points, (int, np.integer)) or angular_points < 2:
        raise ValueError("angular_points must be an integer >= 2.")
    if interpolation_kind not in {"linear", "cubic"}:
        raise ValueError("interpolation_kind must be 'linear' or 'cubic'.")

    grids = _validate_grids(
        grids,
        channel_configs,
        extrapolate_hats=extrapolate_hats,
    )
    state = build_initial_state(
        grids,
        channel_configs,
        initial_state=initial_state,
    )

    history = []
    converged = False
    last_hats = None

    for iteration in range(1, int(max_iterations) + 1):
        state, last_hats, diagnostics = kt_iteration(
            grids=grids,
            channel_configs=channel_configs,
            state=state,
            mixing=mixing,
            angular_points=int(angular_points),
            interpolation_kind=interpolation_kind,
            extrapolate_state=extrapolate_state,
            extrapolate_hats=extrapolate_hats,
        )
        diagnostics = {"iteration": iteration, **diagnostics}
        history.append(diagnostics)

        if verbose:
            print(
                f"iteration {iteration:3d}: "
                f"residual_abs={diagnostics['residual_absolute']:.3e}, "
                f"residual_rel={diagnostics['residual_relative']:.3e}"
            )

        convergence_bound = (
            absolute_tolerance
            + relative_tolerance * diagnostics["scale"]
        )
        if diagnostics["residual_absolute"] <= convergence_bound:
            converged = True
            break

    final_interpolators = build_state_interpolators(
        grids,
        state,
        kind=interpolation_kind,
        extrapolate=extrapolate_state,
    )
    final_hats = compute_hat_values(
        grids,
        channel_configs,
        final_interpolators,
        n_points=int(angular_points),
    )
    final_hat_interpolators = {
        family: kt.interpolate_nested_hat_functions(
            grids[family],
            values,
            kind=interpolation_kind,
            extrapolate=extrapolate_hats,
        )
        for family, values in final_hats.items()
    }

    return {
        "converged": converged,
        "iterations": len(history),
        "history": history,
        "grids": grids,
        "state": state,
        "interpolators": final_interpolators,
        "hats": final_hats,
        "hat_interpolators": final_hat_interpolators,
        "last_iteration_hats": last_hats,
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def run_self_checks():
    """Run a zero-driving fixed-point test through all three families."""
    import kinematics as kin

    s_grid, s_lower, s_upper = make_solver_grid(
        kin.s_decay_min,
        kin.s_decay_max,
        n_points=6,
        edge_fraction=2.0e-3,
    )
    t_grid, t_lower, t_upper = make_solver_grid(
        kin.t_decay_min,
        kin.t_decay_max,
        n_points=6,
        edge_fraction=2.0e-3,
    )
    u_grid, u_lower, u_upper = make_solver_grid(
        kin.u_decay_min,
        kin.u_decay_max,
        n_points=6,
        edge_fraction=2.0e-3,
    )

    zero_phase = lambda x: 0.0
    unit_omega = lambda x: 1.0 + 0.0j

    config_s = channel_config(
        zero_phase,
        unit_omega,
        s_lower,
        s_upper,
        subtraction_coefficients=[0.0],
        n_subtractions=1,
    )
    config_t = channel_config(
        zero_phase,
        unit_omega,
        t_lower,
        t_upper,
        subtraction_coefficients=[0.0],
        n_subtractions=1,
    )
    config_u = channel_config(
        zero_phase,
        unit_omega,
        u_lower,
        u_upper,
        subtraction_coefficients=[0.0],
        n_subtractions=1,
    )

    configs = {
        "M": {"0": {"S0": config_s}},
        "N": {"1/2": {"S0": config_t}},
        "R": {"1/2": {"S0": config_u}},
    }
    result = solve_kt(
        grids={"M": s_grid, "N": t_grid, "R": u_grid},
        channel_configs=configs,
        max_iterations=2,
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        mixing=0.5,
        angular_points=12,
        interpolation_kind="cubic",
        extrapolate_state=False,
        extrapolate_hats=False,
        verbose=False,
    )

    maximum = _maximum_state_norm(result["state"])
    if not result["converged"] or result["iterations"] != 1 or maximum > 1.0e-13:
        raise RuntimeError(
            "kt_solver self-check failed: "
            f"converged={result['converged']}, "
            f"iterations={result['iterations']}, maximum={maximum}."
        )

    return {
        "converged": result["converged"],
        "iterations": result["iterations"],
        "maximum_amplitude": maximum,
    }


__all__ = [
    "ALLOWED_CHANNELS",
    "channel_config",
    "make_solver_grid",
    "build_initial_state",
    "build_state_interpolators",
    "compute_hat_values",
    "kt_iteration",
    "solve_kt",
    "run_self_checks",
]
