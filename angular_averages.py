"""S- and P-wave angular averages for D1 -> D pi pi.

This module implements the conventions in Sections 4--6 of ``D1Dpipi.pdf``:

* longitudinal helicity waves are expanded with ``P_J(z)``;
* transverse helicity waves are expanded with ``P'_J(z)``;
* the reduced waves use the kinematic factors ``X``, ``p`` and ``kappa``;
* the isospin crossing matrices use the ordering
  ``s: (0, 1)`` and ``t/u: (1/2, 3/2)``.

Only J = 0 and J = 1 are retained.  The module reconstructs the invariant
form factors F_A and F_B from single-variable functions and projects the
crossed-channel pieces to obtain the hat functions.

All masses are in GeV and all Mandelstam variables are in GeV**2, matching
``kinematics.py``.

Important
---------
The supplied ``kinematics.py`` uses real square roots.  Consequently these
angular averages are intended for points inside the real decay region and
must not be used for complex contour deformation or general analytic
continuation without replacing the square-root prescription.

Amplitude dictionaries
----------------------
The single-variable functions are supplied as dictionaries.  Each entry may
be either a callable or a constant.  Missing entries are interpreted as zero.

For the s channel::

    M = {
        "0": {"S0": M_0_0},
        "1": {"P0": M_1_0, "Pperp": M_1_perp},
    }

For the t and u channels::

    N = {
        "1/2": {"S0": N_0_0_half,
                 "P0": N_1_0_half,
                 "Pperp": N_1_perp_half},
        "3/2": {"S0": N_0_0_threehalf,
                 "P0": N_1_0_threehalf,
                 "Pperp": N_1_perp_threehalf},
    }

``R`` has the same structure as ``N``.  Array-valued evaluations are used
during Gaussian quadrature, so callable entries should accept NumPy arrays.
"""

from functools import lru_cache
from typing import Callable, Mapping, Tuple

import numpy as np

import kinematics as kin


# ---------------------------------------------------------------------------
# Channel labels and isospin crossing matrices
# ---------------------------------------------------------------------------

S_ISOSPIN = ("0", "1")
DPI_ISOSPIN = ("1/2", "3/2")

SQRT6 = np.sqrt(6.0)

# Row order: target-channel isospin.  Column order: source-channel isospin.
C_S_FROM_T = np.array(
    [
        [SQRT6 / 3.0, 2.0 * SQRT6 / 3.0],
        [-2.0 / 3.0, 2.0 / 3.0],
    ],
    dtype=float,
)

C_S_FROM_U = np.array(
    [
        [SQRT6 / 3.0, 2.0 * SQRT6 / 3.0],
        [2.0 / 3.0, -2.0 / 3.0],
    ],
    dtype=float,
)

C_T_FROM_S = np.array(
    [
        [SQRT6 / 6.0, -1.0],
        [SQRT6 / 6.0, 1.0 / 2.0],
    ],
    dtype=float,
)

C_T_FROM_U = np.array(
    [
        [-1.0 / 3.0, 4.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0],
    ],
    dtype=float,
)

C_U_FROM_S = np.array(
    [
        [SQRT6 / 6.0, 1.0],
        [SQRT6 / 6.0, -1.0 / 2.0],
    ],
    dtype=float,
)

C_U_FROM_T = np.array(
    [
        [-1.0 / 3.0, 4.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

AmplitudeMap = Mapping[str, Mapping[str, object]]
FormFactorCallback = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _as_real_array(x, name):
    array = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real values.")
    return array


def _require_nonzero(x, name, tolerance=1.0e-14):
    if np.any(np.abs(x) <= tolerance):
        raise ValueError(
            f"{name} vanishes at this kinematic endpoint. "
            "Evaluate the angular average at an interior point, for example "
            "threshold + epsilon."
        )


def _component(amplitudes, isospin, wave, x):
    """Evaluate one dictionary entry and broadcast it to the shape of x."""
    x_array = np.asarray(x)

    if amplitudes is None:
        return np.zeros(x_array.shape, dtype=complex)

    channel = amplitudes.get(isospin, {})
    entry = channel.get(wave, 0.0)
    value = entry(x_array) if callable(entry) else entry
    value = np.asarray(value, dtype=complex)

    try:
        return np.broadcast_to(value, x_array.shape)
    except ValueError as error:
        raise ValueError(
            f"Amplitude entry amplitudes[{isospin!r}][{wave!r}] returned "
            f"shape {value.shape}, which cannot be broadcast to {x_array.shape}."
        ) from error


def _form_factor_arrays(form_factors, z):
    F_A, F_B = form_factors(z)
    z = np.asarray(z)

    try:
        F_A = np.broadcast_to(np.asarray(F_A, dtype=complex), z.shape)
        F_B = np.broadcast_to(np.asarray(F_B, dtype=complex), z.shape)
    except ValueError as error:
        raise ValueError(
            "The form-factor callback must return F_A and F_B values that "
            "broadcast to the quadrature grid."
        ) from error

    return F_A, F_B


@lru_cache(maxsize=None)
def gauss_legendre_grid(n_points=80):
    """Return Gauss-Legendre nodes and weights on [-1, 1]."""
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer greater than or equal to 2.")

    z, weights = np.polynomial.legendre.leggauss(int(n_points))
    return z, weights


def _integrate_on_minus_one_one(values, weights):
    return np.sum(weights * np.asarray(values, dtype=complex))


# ---------------------------------------------------------------------------
# Kinematic quantities used by the reduced helicity waves
# ---------------------------------------------------------------------------

def X_s(s):
    """X(s) = sqrt(lambda(s, m_D1^2, m_D^2)) / (2 sqrt(s))."""
    return kin.p_D_s(s)


def E_D1_s(s):
    """D1 energy in the crossed s-channel center-of-mass frame."""
    s = _as_real_array(s, "s")
    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")
    return (s + kin.m_D1**2 - kin.m_D**2) / (2.0 * np.sqrt(s))


def reduced_kappa_s(s):
    """Reduced s-channel factor X(s) q(s) / m_D1^2."""
    return X_s(s) * kin.q_pi_s(s) / kin.m_D1**2


def p_dpi(x):
    """Initial D1-pion momentum in a crossed D-pi channel."""
    x = _as_real_array(x, "D-pi Mandelstam variable")
    if np.any(x <= 0.0):
        raise ValueError("The D-pi Mandelstam variable must be positive.")
    return (
        kin.physical_sqrt(kin.kallen(x, kin.m_D1**2, kin.m_pi**2))
        / (2.0 * np.sqrt(x))
    )


def q_dpi(x):
    """Final D-pion momentum in a crossed D-pi channel."""
    x = _as_real_array(x, "D-pi Mandelstam variable")
    if np.any(x <= 0.0):
        raise ValueError("The D-pi Mandelstam variable must be positive.")
    return (
        kin.physical_sqrt(kin.kallen(x, kin.m_D**2, kin.m_pi**2))
        / (2.0 * np.sqrt(x))
    )


def E_D1_dpi(x):
    """D1 energy in a crossed D-pi center-of-mass frame."""
    x = _as_real_array(x, "D-pi Mandelstam variable")
    if np.any(x <= 0.0):
        raise ValueError("The D-pi Mandelstam variable must be positive.")
    return (x + kin.m_D1**2 - kin.m_pi**2) / (2.0 * np.sqrt(x))


def E_pi_dpi(x):
    """Final-pion energy in a crossed D-pi center-of-mass frame."""
    x = _as_real_array(x, "D-pi Mandelstam variable")
    if np.any(x <= 0.0):
        raise ValueError("The D-pi Mandelstam variable must be positive.")
    return (x + kin.m_pi**2 - kin.m_D**2) / (2.0 * np.sqrt(x))


def reduced_kappa_dpi(x):
    """Reduced t/u-channel factor p(x) q(x) / m_D1^2."""
    return p_dpi(x) * q_dpi(x) / kin.m_D1**2


def alpha_dpi(x, z):
    """Alpha factor used in the t- and u-channel reconstruction."""
    x = _as_real_array(x, "D-pi Mandelstam variable")
    z = _as_real_array(z, "D-pi scattering angle")
    p = p_dpi(x)
    _require_nonzero(p, "p_dpi")
    return (
        p * E_pi_dpi(x) + E_D1_dpi(x) * q_dpi(x) * z
    ) / (p * np.sqrt(x))


# ---------------------------------------------------------------------------
# Inverse angle maps: (s, t, u) -> z_s, z_t, z_u
# ---------------------------------------------------------------------------

def z_s_from_stu(s, t, u):
    """Return z_s = (u - t) / [sqrt(lambda_1 lambda_2) / s]."""
    s, t, u = np.broadcast_arrays(
        _as_real_array(s, "s"),
        _as_real_array(t, "t"),
        _as_real_array(u, "u"),
    )
    denominator = (
        kin.physical_sqrt(kin.kallen(s, kin.m_D1**2, kin.m_D**2))
        * kin.physical_sqrt(kin.kallen(s, kin.m_pi**2, kin.m_pi**2))
        / s
    )
    _require_nonzero(denominator, "the z_s denominator")
    return (u - t) / denominator


def z_t_from_stu(s, t, u):
    """Return z_t using the angle convention of ``kinematics.py``."""
    s, t, u = np.broadcast_arrays(
        _as_real_array(s, "s"),
        _as_real_array(t, "t"),
        _as_real_array(u, "u"),
    )
    denominator = (
        kin.physical_sqrt(kin.kallen(t, kin.m_D1**2, kin.m_pi**2))
        * kin.physical_sqrt(kin.kallen(t, kin.m_D**2, kin.m_pi**2))
        / t
    )
    _require_nonzero(denominator, "the z_t denominator")
    return (s - u + kin.Delta_Dpi / t) / denominator


def z_u_from_stu(s, t, u):
    """Return z_u using the angle convention of ``kinematics.py``."""
    s, t, u = np.broadcast_arrays(
        _as_real_array(s, "s"),
        _as_real_array(t, "t"),
        _as_real_array(u, "u"),
    )
    denominator = (
        kin.physical_sqrt(kin.kallen(u, kin.m_D1**2, kin.m_pi**2))
        * kin.physical_sqrt(kin.kallen(u, kin.m_D**2, kin.m_pi**2))
        / u
    )
    _require_nonzero(denominator, "the z_u denominator")
    return (s - t + kin.Delta_Dpi / u) / denominator


# ---------------------------------------------------------------------------
# S+P reconstruction of F_A and F_B
# ---------------------------------------------------------------------------

def reconstruct_s_form_factors(s, z_s, M, isospin):
    """Reconstruct the s-channel F_A and F_B through J <= 1."""
    if isospin not in S_ISOSPIN:
        raise ValueError(f"s-channel isospin must be one of {S_ISOSPIN}.")

    s = _as_real_array(s, "s")
    z_s = _as_real_array(z_s, "z_s")
    root_s = np.sqrt(s)
    X = X_s(s)
    sigma = kin.sigma_pi(s)
    kappa = reduced_kappa_s(s)

    _require_nonzero(X, "X_s")
    _require_nonzero(sigma, "sigma_pi")

    M00 = _component(M, isospin, "S0", s)
    M10 = _component(M, isospin, "P0", s)
    M1perp = _component(M, isospin, "Pperp", s)

    F_A = (
        kin.m_D1**2 / root_s * (M00 + 3.0 * kappa * z_s * M10)
        + 3.0 * kin.m_D1 * E_D1_s(s) * z_s
        / (root_s * X) * M1perp
    )
    F_B = 3.0 * kin.m_D1 / (sigma * root_s) * M1perp

    return F_A, F_B


def reconstruct_t_form_factors(t, z_t, N, isospin):
    """Reconstruct the t-channel F_A and F_B through J <= 1."""
    if isospin not in DPI_ISOSPIN:
        raise ValueError(f"t-channel isospin must be one of {DPI_ISOSPIN}.")

    t = _as_real_array(t, "t")
    z_t = _as_real_array(z_t, "z_t")
    root_t = np.sqrt(t)
    q = q_dpi(t)
    kappa = reduced_kappa_dpi(t)
    alpha = alpha_dpi(t, z_t)

    _require_nonzero(q, "q_dpi")

    N00 = _component(N, isospin, "S0", t)
    N10 = _component(N, isospin, "P0", t)
    N1perp = _component(N, isospin, "Pperp", t)
    longitudinal = N00 + 3.0 * kappa * z_t * N10

    F_A = (
        -kin.m_D1**2 / (2.0 * root_t) * longitudinal
        + 3.0 * kin.m_D1 / (2.0 * q) * (1.0 + alpha) * N1perp
    )
    F_B = (
        +kin.m_D1**2 / (2.0 * root_t) * longitudinal
        + 3.0 * kin.m_D1 / (2.0 * q) * (1.0 - alpha) * N1perp
    )

    return F_A, F_B


def reconstruct_u_form_factors(u, z_u, R, isospin):
    """Reconstruct the u-channel F_A and F_B through J <= 1."""
    if isospin not in DPI_ISOSPIN:
        raise ValueError(f"u-channel isospin must be one of {DPI_ISOSPIN}.")

    u = _as_real_array(u, "u")
    z_u = _as_real_array(z_u, "z_u")
    root_u = np.sqrt(u)
    q = q_dpi(u)
    kappa = reduced_kappa_dpi(u)
    alpha = alpha_dpi(u, z_u)

    _require_nonzero(q, "q_dpi")

    R00 = _component(R, isospin, "S0", u)
    R10 = _component(R, isospin, "P0", u)
    R1perp = _component(R, isospin, "Pperp", u)
    longitudinal = R00 + 3.0 * kappa * z_u * R10

    F_A = (
        -kin.m_D1**2 / (2.0 * root_u) * longitudinal
        + 3.0 * kin.m_D1 / (2.0 * q) * (1.0 + alpha) * R1perp
    )
    F_B = (
        -kin.m_D1**2 / (2.0 * root_u) * longitudinal
        + 3.0 * kin.m_D1 / (2.0 * q) * (alpha - 1.0) * R1perp
    )

    return F_A, F_B


# ---------------------------------------------------------------------------
# Crossed-channel invariant form factors
# ---------------------------------------------------------------------------

def _matrix_combination(matrix, row, source_values):
    F_A = 0.0j
    F_B = 0.0j
    for column, (source_F_A, source_F_B) in enumerate(source_values):
        F_A = F_A + matrix[row, column] * source_F_A
        F_B = F_B + matrix[row, column] * source_F_B
    return F_A, F_B


def crossed_s_form_factors(s, z_s, N, R, isospin):
    """Crossed t+u contribution to fixed-s-isospin form factors."""
    if isospin not in S_ISOSPIN:
        raise ValueError(f"s-channel isospin must be one of {S_ISOSPIN}.")

    t = kin.t_of_s_z(s, z_s)
    u = kin.u_of_s_z(s, z_s)
    z_t = z_t_from_stu(s, t, u)
    z_u = z_u_from_stu(s, t, u)

    t_values = [
        reconstruct_t_form_factors(t, z_t, N, source_isospin)
        for source_isospin in DPI_ISOSPIN
    ]
    u_values = [
        reconstruct_u_form_factors(u, z_u, R, source_isospin)
        for source_isospin in DPI_ISOSPIN
    ]

    row = S_ISOSPIN.index(isospin)
    F_A_t, F_B_t = _matrix_combination(C_S_FROM_T, row, t_values)
    F_A_u, F_B_u = _matrix_combination(C_S_FROM_U, row, u_values)
    return F_A_t + F_A_u, F_B_t + F_B_u


def crossed_t_form_factors(t, z_t, M, R, isospin):
    """Crossed s+u contribution to fixed-t-isospin form factors."""
    if isospin not in DPI_ISOSPIN:
        raise ValueError(f"t-channel isospin must be one of {DPI_ISOSPIN}.")

    s = kin.s_of_t_z(t, z_t)
    u = kin.u_of_t_z(t, z_t)
    z_s = z_s_from_stu(s, t, u)
    z_u = z_u_from_stu(s, t, u)

    s_values = [
        reconstruct_s_form_factors(s, z_s, M, source_isospin)
        for source_isospin in S_ISOSPIN
    ]
    u_values = [
        reconstruct_u_form_factors(u, z_u, R, source_isospin)
        for source_isospin in DPI_ISOSPIN
    ]

    row = DPI_ISOSPIN.index(isospin)
    F_A_s, F_B_s = _matrix_combination(C_T_FROM_S, row, s_values)
    F_A_u, F_B_u = _matrix_combination(C_T_FROM_U, row, u_values)
    return F_A_s + F_A_u, F_B_s + F_B_u


def crossed_u_form_factors(u, z_u, M, N, isospin):
    """Crossed s+t contribution to fixed-u-isospin form factors."""
    if isospin not in DPI_ISOSPIN:
        raise ValueError(f"u-channel isospin must be one of {DPI_ISOSPIN}.")

    s = kin.s_of_u_z(u, z_u)
    t = kin.t_of_u_z(u, z_u)
    z_s = z_s_from_stu(s, t, u)
    z_t = z_t_from_stu(s, t, u)

    s_values = [
        reconstruct_s_form_factors(s, z_s, M, source_isospin)
        for source_isospin in S_ISOSPIN
    ]
    t_values = [
        reconstruct_t_form_factors(t, z_t, N, source_isospin)
        for source_isospin in DPI_ISOSPIN
    ]

    row = DPI_ISOSPIN.index(isospin)
    F_A_s, F_B_s = _matrix_combination(C_U_FROM_S, row, s_values)
    F_A_t, F_B_t = _matrix_combination(C_U_FROM_T, row, t_values)
    return F_A_s + F_A_t, F_B_s + F_B_t


# ---------------------------------------------------------------------------
# Generic S/P projections in the three channels
# ---------------------------------------------------------------------------

def project_s_sp(s, form_factors, n_points=80):
    """Project an s-channel F_A,F_B callback onto reduced S/P waves."""
    s = float(_as_real_array(s, "s"))
    z, weights = gauss_legendre_grid(n_points)
    F_A, F_B = _form_factor_arrays(form_factors, z)

    X = X_s(s)
    sigma = kin.sigma_pi(s)
    kappa = reduced_kappa_s(s)
    _require_nonzero(X, "X_s")
    _require_nonzero(kappa, "reduced_kappa_s")

    root_s = np.sqrt(s)
    H_0 = root_s / kin.m_D1**2 * (
        X * F_A - E_D1_s(s) * sigma * z * F_B
    )
    H_perp = (
        root_s * sigma / kin.m_D1
        * np.sqrt(1.0 - z**2) * F_B
    )

    return {
        "S0": _integrate_on_minus_one_one(H_0, weights) / (2.0 * X),
        "P0": _integrate_on_minus_one_one(z * H_0, weights)
        / (2.0 * X * kappa),
        "Pperp": _integrate_on_minus_one_one(
            np.sqrt(1.0 - z**2) * H_perp,
            weights,
        ) / 4.0,
    }


def project_t_sp(t, form_factors, n_points=80):
    """Project a t-channel F_A,F_B callback onto reduced S/P waves."""
    t = float(_as_real_array(t, "t"))
    z, weights = gauss_legendre_grid(n_points)
    F_A, F_B = _form_factor_arrays(form_factors, z)

    p = p_dpi(t)
    q = q_dpi(t)
    kappa = reduced_kappa_dpi(t)
    _require_nonzero(p, "p_dpi")
    _require_nonzero(kappa, "reduced_kappa_dpi")

    F_plus = F_A + F_B
    F_minus = F_A - F_B
    H_0 = (
        (p * E_pi_dpi(t) + E_D1_dpi(t) * q * z) * F_plus
        - p * np.sqrt(t) * F_minus
    ) / kin.m_D1**2
    H_perp = q / kin.m_D1 * np.sqrt(1.0 - z**2) * F_plus

    return {
        "S0": _integrate_on_minus_one_one(H_0, weights) / (2.0 * p),
        "P0": _integrate_on_minus_one_one(z * H_0, weights)
        / (2.0 * p * kappa),
        "Pperp": _integrate_on_minus_one_one(
            np.sqrt(1.0 - z**2) * H_perp,
            weights,
        ) / 4.0,
    }


def project_u_sp(u, form_factors, n_points=80):
    """Project a u-channel F_A,F_B callback onto reduced S/P waves."""
    u = float(_as_real_array(u, "u"))
    z, weights = gauss_legendre_grid(n_points)
    F_A, F_B = _form_factor_arrays(form_factors, z)

    p = p_dpi(u)
    q = q_dpi(u)
    kappa = reduced_kappa_dpi(u)
    _require_nonzero(p, "p_dpi")
    _require_nonzero(kappa, "reduced_kappa_dpi")

    F_plus = F_A + F_B
    F_minus = F_A - F_B
    H_0 = (
        (p * E_pi_dpi(u) + E_D1_dpi(u) * q * z) * F_minus
        - p * np.sqrt(u) * F_plus
    ) / kin.m_D1**2
    H_perp = q / kin.m_D1 * np.sqrt(1.0 - z**2) * F_minus

    return {
        "S0": _integrate_on_minus_one_one(H_0, weights) / (2.0 * p),
        "P0": _integrate_on_minus_one_one(z * H_0, weights)
        / (2.0 * p * kappa),
        "Pperp": _integrate_on_minus_one_one(
            np.sqrt(1.0 - z**2) * H_perp,
            weights,
        ) / 4.0,
    }


# ---------------------------------------------------------------------------
# Hat functions
# ---------------------------------------------------------------------------

def _vectorize_nested_hat(x, scalar_function):
    x = _as_real_array(x, "Mandelstam variable")
    if x.ndim == 0:
        return scalar_function(float(x))

    first = scalar_function(float(x.flat[0]))
    result = {
        isospin: {
            wave: np.empty(x.shape, dtype=complex)
            for wave in waves
        }
        for isospin, waves in first.items()
    }

    for index in np.ndindex(x.shape):
        local = first if index == tuple(0 for _ in x.shape) else scalar_function(
            float(x[index])
        )
        for isospin, waves in local.items():
            for wave, value in waves.items():
                result[isospin][wave][index] = value

    return result


def hat_s(s, N, R, n_points=80):
    """Return the retained s-channel hats M-hat through S and P waves.

    The returned structure is::

        {
            "0": {"S0": Mhat_0_0},
            "1": {"P0": Mhat_1_0, "Pperp": Mhat_1_perp},
        }
    """

    def scalar(s_value):
        projected = {}
        for isospin in S_ISOSPIN:
            projected[isospin] = project_s_sp(
                s_value,
                lambda z, iso=isospin: crossed_s_form_factors(
                    s_value, z, N, R, iso
                ),
                n_points=n_points,
            )

        return {
            "0": {"S0": projected["0"]["S0"]},
            "1": {
                "P0": projected["1"]["P0"],
                "Pperp": projected["1"]["Pperp"],
            },
        }

    return _vectorize_nested_hat(s, scalar)


def hat_t(t, M, R, n_points=80):
    """Return t-channel hats N-hat for I=1/2,3/2 and J <= 1."""

    def scalar(t_value):
        return {
            isospin: project_t_sp(
                t_value,
                lambda z, iso=isospin: crossed_t_form_factors(
                    t_value, z, M, R, iso
                ),
                n_points=n_points,
            )
            for isospin in DPI_ISOSPIN
        }

    return _vectorize_nested_hat(t, scalar)


def hat_u(u, M, N, n_points=80):
    """Return u-channel hats R-hat for I=1/2,3/2 and J <= 1."""

    def scalar(u_value):
        return {
            isospin: project_u_sp(
                u_value,
                lambda z, iso=isospin: crossed_u_form_factors(
                    u_value, z, M, N, iso
                ),
                n_points=n_points,
            )
            for isospin in DPI_ISOSPIN
        }

    return _vectorize_nested_hat(u, scalar)


# ---------------------------------------------------------------------------
# Internal consistency checks
# ---------------------------------------------------------------------------

def validate_crossing_matrices(atol=1.0e-14):
    """Check that the forward and inverse crossing matrices are consistent."""
    identity = np.eye(2)
    checks = (
        C_T_FROM_S @ C_S_FROM_T,
        C_U_FROM_S @ C_S_FROM_U,
        C_T_FROM_U @ C_U_FROM_T,
    )
    if not all(np.allclose(value, identity, atol=atol, rtol=0.0) for value in checks):
        raise RuntimeError("The isospin crossing matrices are inconsistent.")
    return True


def run_self_checks(n_points=100, atol=2.0e-11):
    """Numerically check kinematic maps and S/P projection normalizations."""
    validate_crossing_matrices()

    constant = lambda value: (
        lambda x: value + 0.0j * np.asarray(x)
    )

    trial = {
        "0": {
            "S0": constant(1.2 - 0.3j),
            "P0": constant(-0.4 + 0.2j),
            "Pperp": constant(0.7 + 0.1j),
        }
    }
    expected = {
        "S0": 1.2 - 0.3j,
        "P0": -0.4 + 0.2j,
        "Pperp": 0.7 + 0.1j,
    }

    s = 0.5 * (kin.s_decay_min + kin.s_decay_max)
    t = 0.5 * (kin.t_decay_min + kin.t_decay_max)
    u = 0.5 * (kin.u_decay_min + kin.u_decay_max)

    recovered_s = project_s_sp(
        s,
        lambda z: reconstruct_s_form_factors(s, z, trial, "0"),
        n_points=n_points,
    )

    trial_dpi = {"1/2": trial["0"]}
    recovered_t = project_t_sp(
        t,
        lambda z: reconstruct_t_form_factors(t, z, trial_dpi, "1/2"),
        n_points=n_points,
    )
    recovered_u = project_u_sp(
        u,
        lambda z: reconstruct_u_form_factors(u, z, trial_dpi, "1/2"),
        n_points=n_points,
    )

    errors = {
        "s": max(abs(recovered_s[key] - expected[key]) for key in expected),
        "t": max(abs(recovered_t[key] - expected[key]) for key in expected),
        "u": max(abs(recovered_u[key] - expected[key]) for key in expected),
    }

    if max(errors.values()) > atol:
        raise RuntimeError(f"S/P reconstruction self-check failed: {errors}")

    # Forward maps must satisfy s + t + u = Sigma.
    z_test = np.array([-0.7, 0.0, 0.8])
    t_test = kin.t_of_s_z(s, z_test)
    u_test = kin.u_of_s_z(s, z_test)
    if not np.allclose(s + t_test + u_test, kin.Sigma, atol=atol, rtol=0.0):
        raise RuntimeError("The Mandelstam-variable map failed s+t+u=Sigma.")

    return errors


__all__ = [
    "S_ISOSPIN",
    "DPI_ISOSPIN",
    "C_S_FROM_T",
    "C_S_FROM_U",
    "C_T_FROM_S",
    "C_T_FROM_U",
    "C_U_FROM_S",
    "C_U_FROM_T",
    "gauss_legendre_grid",
    "X_s",
    "E_D1_s",
    "reduced_kappa_s",
    "p_dpi",
    "q_dpi",
    "E_D1_dpi",
    "E_pi_dpi",
    "reduced_kappa_dpi",
    "alpha_dpi",
    "z_s_from_stu",
    "z_t_from_stu",
    "z_u_from_stu",
    "reconstruct_s_form_factors",
    "reconstruct_t_form_factors",
    "reconstruct_u_form_factors",
    "crossed_s_form_factors",
    "crossed_t_form_factors",
    "crossed_u_form_factors",
    "project_s_sp",
    "project_t_sp",
    "project_u_sp",
    "hat_s",
    "hat_t",
    "hat_u",
    "validate_crossing_matrices",
    "run_self_checks",
]
