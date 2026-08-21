"""Reconstruct D1 -> D pi pi amplitudes for six physical charge channels.

The module follows the conventions of ``D1Dpipi.pdf``:

    M = epsilon_mu [P^mu F_A + Q^mu F_B] / m_D1,

with

    s = (p1 + p2)^2,
    t = (pD + p1)^2,
    u = (pD + p2)^2.

The KT solution supplies the single-variable functions M, N and R carrying
the s-, t- and u-channel right-hand cuts.  This file first reconstructs the
fixed-s-isospin form factors and then applies the physical-process matrix S.

The six canonical process names preserve the pion ordering p1, p2 used in the
theory notes.  For example, in ``D10_to_D0_pip_pim``, p1 belongs to pi+ and p2
belongs to pi-.

All masses are in GeV and all Mandelstam variables are in GeV**2.  The current
implementation is restricted to the real decay region, consistently with
``kinematics.py`` and ``angular_averages.py``.
"""

from typing import Mapping

import numpy as np

import angular_averages as angular
import kinematics as kin


# ---------------------------------------------------------------------------
# Six physical processes and the s-channel isospin matrix
# ---------------------------------------------------------------------------

PROCESS_ORDER = (
    "D10_to_D0_pip_pim",
    "D10_to_D0_pi0_pi0",
    "D10_to_Dp_pim_pi0",
    "D1p_to_Dp_pip_pim",
    "D1p_to_Dp_pi0_pi0",
    "D1p_to_D0_pip_pi0",
)

PROCESS_INFO = {
    "D10_to_D0_pip_pim": {
        "initial": "D1^0",
        "D": "D^0",
        "pion1": "pi+",
        "pion2": "pi-",
        "identical_pions": False,
        "notation": "M^0_{+-}",
    },
    "D10_to_D0_pi0_pi0": {
        "initial": "D1^0",
        "D": "D^0",
        "pion1": "pi0",
        "pion2": "pi0",
        "identical_pions": True,
        "notation": "M^0_{00}",
    },
    "D10_to_Dp_pim_pi0": {
        "initial": "D1^0",
        "D": "D+",
        "pion1": "pi-",
        "pion2": "pi0",
        "identical_pions": False,
        "notation": "M^+_{-0}",
    },
    "D1p_to_Dp_pip_pim": {
        "initial": "D1+",
        "D": "D+",
        "pion1": "pi+",
        "pion2": "pi-",
        "identical_pions": False,
        "notation": "M^+_{+-}",
    },
    "D1p_to_Dp_pi0_pi0": {
        "initial": "D1+",
        "D": "D+",
        "pion1": "pi0",
        "pion2": "pi0",
        "identical_pions": True,
        "notation": "M^+_{00}",
    },
    "D1p_to_D0_pip_pi0": {
        "initial": "D1+",
        "D": "D0",
        "pion1": "pi+",
        "pion2": "pi0",
        "identical_pions": False,
        "notation": "M^0_{+0}",
    },
}

# Column order: fixed s-channel isospin (I_s=0, I_s=1).
PHYSICAL_S_MATRIX = np.array(
    [
        [1.0 / np.sqrt(6.0), -1.0 / 2.0],
        [1.0 / np.sqrt(6.0), 0.0],
        [0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), 1.0 / 2.0],
        [1.0 / np.sqrt(6.0), 0.0],
        [0.0, -1.0 / np.sqrt(2.0)],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Input and solution helpers
# ---------------------------------------------------------------------------

def _real_array(x, name):
    array = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values.")
    return array


def prepare_invariants(s, t, u=None, sum_tolerance=2.0e-9):
    """Broadcast and validate s,t,u; calculate u=Sigma-s-t if omitted."""
    s = _real_array(s, "s")
    t = _real_array(t, "t")

    if u is None:
        s, t = np.broadcast_arrays(s, t)
        u = kin.Sigma - s - t
    else:
        u = _real_array(u, "u")
        s, t, u = np.broadcast_arrays(s, t, u)

    if np.any(s <= 0.0) or np.any(t <= 0.0) or np.any(u <= 0.0):
        raise ValueError("s, t and u must be positive in the real decay region.")

    mismatch = np.max(np.abs(s + t + u - kin.Sigma))
    scale = max(1.0, abs(kin.Sigma))
    if mismatch > sum_tolerance * scale:
        raise ValueError(
            "The Mandelstam variables do not satisfy s+t+u=Sigma. "
            f"Maximum mismatch: {mismatch}."
        )

    return s, t, u


def _extract_amplitudes(solution):
    """Accept a solve_kt result or a direct mapping of callable M,N,R."""
    if not isinstance(solution, Mapping):
        raise TypeError("solution must be a mapping.")

    if "interpolators" in solution:
        amplitudes = solution["interpolators"]
    else:
        amplitudes = solution

    if not isinstance(amplitudes, Mapping):
        raise TypeError("solution['interpolators'] must be a mapping.")

    return {
        "M": amplitudes.get("M", {}),
        "N": amplitudes.get("N", {}),
        "R": amplitudes.get("R", {}),
    }


def _validate_process(process):
    if process not in PROCESS_INFO:
        raise ValueError(
            f"Unknown process {process!r}.  Choose one of {PROCESS_ORDER}."
        )
    return PROCESS_ORDER.index(process)


def _pair_add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _matrix_pair_combination(matrix, row, source_pairs):
    F_A = 0.0j
    F_B = 0.0j
    for column, pair in enumerate(source_pairs):
        F_A = F_A + matrix[row, column] * pair[0]
        F_B = F_B + matrix[row, column] * pair[1]
    return F_A, F_B


# ---------------------------------------------------------------------------
# M/N/R channel reconstruction
# ---------------------------------------------------------------------------

def reconstruct_channel_form_factors(s, t, u, solution):
    """Reconstruct the separate M-, N- and R-channel F_A,F_B pieces.

    Returns
    -------
    dict
        ``pieces['M']['0']`` is ``(F_A, F_B)`` for the direct s-channel
        I_s=0 piece.  N and R use isospin keys ``'1/2'`` and ``'3/2'``.
    """
    s, t, u = prepare_invariants(s, t, u)
    amplitudes = _extract_amplitudes(solution)

    z_s = angular.z_s_from_stu(s, t, u)
    z_t = angular.z_t_from_stu(s, t, u)
    z_u = angular.z_u_from_stu(s, t, u)

    return {
        "M": {
            isospin: angular.reconstruct_s_form_factors(
                s,
                z_s,
                amplitudes["M"],
                isospin,
            )
            for isospin in angular.S_ISOSPIN
        },
        "N": {
            isospin: angular.reconstruct_t_form_factors(
                t,
                z_t,
                amplitudes["N"],
                isospin,
            )
            for isospin in angular.DPI_ISOSPIN
        },
        "R": {
            isospin: angular.reconstruct_u_form_factors(
                u,
                z_u,
                amplitudes["R"],
                isospin,
            )
            for isospin in angular.DPI_ISOSPIN
        },
    }


def all_fixed_isospin_form_factors(s, t, u, solution):
    """Return the fully reconstructed I_s=0 and I_s=1 form factors."""
    pieces = reconstruct_channel_form_factors(s, t, u, solution)
    t_pairs = [pieces["N"][isospin] for isospin in angular.DPI_ISOSPIN]
    u_pairs = [pieces["R"][isospin] for isospin in angular.DPI_ISOSPIN]

    result = {}
    for row, isospin in enumerate(angular.S_ISOSPIN):
        t_crossed = _matrix_pair_combination(
            angular.C_S_FROM_T,
            row,
            t_pairs,
        )
        u_crossed = _matrix_pair_combination(
            angular.C_S_FROM_U,
            row,
            u_pairs,
        )
        result[isospin] = _pair_add(
            pieces["M"][isospin],
            _pair_add(t_crossed, u_crossed),
        )

    return result


def fixed_isospin_form_factors(s, t, u, solution, isospin):
    """Return fully reconstructed ``(F_A,F_B)`` for I_s=0 or I_s=1."""
    if isospin not in angular.S_ISOSPIN:
        raise ValueError(f"isospin must be one of {angular.S_ISOSPIN}.")
    return all_fixed_isospin_form_factors(s, t, u, solution)[isospin]


# ---------------------------------------------------------------------------
# Physical charge-channel reconstruction
# ---------------------------------------------------------------------------

def physical_form_factors_from_fixed(fixed_form_factors, process):
    """Apply the physical S matrix to already reconstructed I_s form factors."""
    row = _validate_process(process)
    try:
        fixed_pairs = [fixed_form_factors["0"], fixed_form_factors["1"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "fixed_form_factors must contain keys '0' and '1', each holding "
            "an (F_A,F_B) pair."
        ) from error

    return _matrix_pair_combination(
        PHYSICAL_S_MATRIX,
        row,
        fixed_pairs,
    )


def physical_form_factors(s, t, u, solution, process):
    """Return physical ``(F_A,F_B)`` for one of the six charge processes."""
    fixed = all_fixed_isospin_form_factors(s, t, u, solution)
    return physical_form_factors_from_fixed(fixed, process)


def all_physical_form_factors(s, t, u, solution):
    """Return physical ``(F_A,F_B)`` pairs for all six charge processes."""
    fixed = all_fixed_isospin_form_factors(s, t, u, solution)
    return {
        process: physical_form_factors_from_fixed(fixed, process)
        for process in PROCESS_ORDER
    }


# ---------------------------------------------------------------------------
# Helicity amplitudes in the s-channel center-of-mass frame
# ---------------------------------------------------------------------------

def helicity_amplitudes_s(s, t, u, F_A, F_B, angle_tolerance=2.0e-10):
    """Construct H_0, H_+, H_-, H_perp at azimuth phi_s=0.

    The returned amplitudes satisfy

        H_perp = (H_plus - H_minus)/sqrt(2),
        H_minus = -H_plus.
    """
    s, t, u = prepare_invariants(s, t, u)
    F_A = np.asarray(F_A, dtype=complex)
    F_B = np.asarray(F_B, dtype=complex)

    try:
        F_A = np.broadcast_to(F_A, s.shape)
        F_B = np.broadcast_to(F_B, s.shape)
    except ValueError as error:
        raise ValueError("F_A and F_B must broadcast to the s,t,u shape.") from error

    z_s = angular.z_s_from_stu(s, t, u)
    if np.any(np.abs(z_s) > 1.0 + angle_tolerance):
        raise ValueError("The supplied point lies outside the real decay angular range.")
    z_s = np.clip(z_s, -1.0, 1.0)
    sin_theta = np.sqrt(np.clip(1.0 - z_s**2, 0.0, None))

    root_s = np.sqrt(s)
    X = angular.X_s(s)
    sigma = kin.sigma_pi(s)
    E_D1 = angular.E_D1_s(s)

    H_0 = root_s / kin.m_D1**2 * (
        X * F_A - E_D1 * sigma * z_s * F_B
    )
    transverse_base = root_s * sigma / kin.m_D1 * sin_theta * F_B
    H_plus = transverse_base / np.sqrt(2.0)
    H_minus = -H_plus
    H_perp = transverse_base

    return {
        "H0": H_0,
        "Hplus": H_plus,
        "Hminus": H_minus,
        "Hperp": H_perp,
    }


def physical_helicity_amplitudes(s, t, u, solution, process):
    """Reconstruct one physical process and return its helicity amplitudes."""
    F_A, F_B = physical_form_factors(
        s,
        t,
        u,
        solution,
        process,
    )
    return helicity_amplitudes_s(s, t, u, F_A, F_B)


def all_physical_helicity_amplitudes(s, t, u, solution):
    """Return helicity amplitudes for all six physical processes."""
    form_factors = all_physical_form_factors(s, t, u, solution)
    return {
        process: helicity_amplitudes_s(
            s,
            t,
            u,
            form_factors[process][0],
            form_factors[process][1],
        )
        for process in PROCESS_ORDER
    }


# ---------------------------------------------------------------------------
# Internal consistency checks
# ---------------------------------------------------------------------------

def run_self_checks(atol=2.0e-11):
    """Check charge coefficients, Bose parity and helicity normalization."""
    errors = {}

    # Direct check of the six physical S-matrix rows.
    fixed_test = {
        "0": (1.2 - 0.1j, -0.3 + 0.2j),
        "1": (0.4 + 0.5j, 0.7 - 0.2j),
    }
    for row, process in enumerate(PROCESS_ORDER):
        numerical = physical_form_factors_from_fixed(fixed_test, process)
        expected_A = (
            PHYSICAL_S_MATRIX[row, 0] * fixed_test["0"][0]
            + PHYSICAL_S_MATRIX[row, 1] * fixed_test["1"][0]
        )
        expected_B = (
            PHYSICAL_S_MATRIX[row, 0] * fixed_test["0"][1]
            + PHYSICAL_S_MATRIX[row, 1] * fixed_test["1"][1]
        )
        errors[f"S_matrix_{row}"] = max(
            abs(numerical[0] - expected_A),
            abs(numerical[1] - expected_B),
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
    z = 0.31
    t = kin.t_of_s_z(s, z)
    u = kin.u_of_s_z(s, z)

    fixed = all_fixed_isospin_form_factors(s, t, u, toy_solution)
    fixed_swapped = all_fixed_isospin_form_factors(s, u, t, toy_solution)

    errors["I0_FA_even"] = abs(fixed["0"][0] - fixed_swapped["0"][0])
    errors["I0_FB_odd"] = abs(fixed["0"][1] + fixed_swapped["0"][1])
    errors["I1_FA_odd"] = abs(fixed["1"][0] + fixed_swapped["1"][0])
    errors["I1_FB_even"] = abs(fixed["1"][1] - fixed_swapped["1"][1])

    F_A, F_B = physical_form_factors(
        s,
        t,
        u,
        toy_solution,
        "D10_to_D0_pip_pim",
    )
    helicity = helicity_amplitudes_s(s, t, u, F_A, F_B)
    errors["Hminus_sign"] = abs(helicity["Hminus"] + helicity["Hplus"])
    errors["Hperp_definition"] = abs(
        helicity["Hperp"]
        - (helicity["Hplus"] - helicity["Hminus"]) / np.sqrt(2.0)
    )

    # The two neutral-pion rows have identical isospin coefficients.
    neutral_0 = physical_form_factors_from_fixed(
        fixed_test,
        "D10_to_D0_pi0_pi0",
    )
    neutral_p = physical_form_factors_from_fixed(
        fixed_test,
        "D1p_to_Dp_pi0_pi0",
    )
    errors["neutral_rows"] = max(
        abs(neutral_0[0] - neutral_p[0]),
        abs(neutral_0[1] - neutral_p[1]),
    )

    maximum = max(errors.values())
    if maximum > atol:
        raise RuntimeError(f"amplitude self-check failed: {errors}")

    return errors


__all__ = [
    "PROCESS_ORDER",
    "PROCESS_INFO",
    "PHYSICAL_S_MATRIX",
    "prepare_invariants",
    "reconstruct_channel_form_factors",
    "all_fixed_isospin_form_factors",
    "fixed_isospin_form_factors",
    "physical_form_factors_from_fixed",
    "physical_form_factors",
    "all_physical_form_factors",
    "helicity_amplitudes_s",
    "physical_helicity_amplitudes",
    "all_physical_helicity_amplitudes",
    "run_self_checks",
]
