"""Generic one-channel Khuri-Treiman dispersive updates.

The module implements

    F(x) = Omega(x) [P(x) + D(x) + T(x)]

with

             (x - x0)^n   / cutoff
    D(x) =  ------------- | dx' sin(delta(x')) hatF(x')
                  pi       / threshold
              -------------------------------------------------- .
              (x' - x0)^n |Omega(x')| (x' - x)

``x0`` is the subtraction point, ``P`` is the subtraction polynomial,
and the optional ``T`` is a user-supplied tail correction inside the Omnes
brackets.  Principal-value integrals are evaluated with SciPy's Cauchy-weight
quadrature when x lies inside the integration interval.

This file is channel independent: the same functions can be used for every
S wave, longitudinal P wave and transverse P wave.  Channel coupling enters
through the supplied hat-function callback.

Conventions
-----------
* Mandelstam variables and integration limits use the same units.
* Phase shifts are in radians.
* ``omega(x)`` and ``hat_function(x)`` may be complex.
* The dispersive integral is cut off at the finite value ``cutoff``.  Physics
  above the cutoff is omitted unless ``tail_correction`` is supplied.
* The external evaluation point x is real in this prototype.
"""

from typing import Mapping

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, interp1d


# ---------------------------------------------------------------------------
# Scalar-value and input helpers
# ---------------------------------------------------------------------------

def _finite_real_array(x, name):
    array = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real values.")
    return array


def _scalar_complex(function, x, name):
    value = function(float(x)) if callable(function) else function
    value = np.asarray(value)

    if value.size != 1:
        raise ValueError(f"{name} must return exactly one value for scalar input.")

    result = complex(value.reshape(-1)[0])
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} returned a non-finite value at x={x}.")
    return result


def _scalar_real(function, x, name, imaginary_tolerance=1.0e-12):
    value = _scalar_complex(function, x, name)
    if abs(value.imag) > imaginary_tolerance:
        raise ValueError(
            f"{name} must be real, but returned {value} at x={x}."
        )
    return float(value.real)


def _validate_interval(threshold, cutoff):
    threshold = float(threshold)
    cutoff = float(cutoff)

    if not np.isfinite(threshold) or not np.isfinite(cutoff):
        raise ValueError("threshold and cutoff must be finite.")
    if cutoff <= threshold:
        raise ValueError("cutoff must be greater than threshold.")

    return threshold, cutoff


def _validate_subtractions(n_subtractions, subtraction_point, threshold, cutoff):
    if not isinstance(n_subtractions, (int, np.integer)):
        raise TypeError("n_subtractions must be an integer.")
    if n_subtractions < 0:
        raise ValueError("n_subtractions must be non-negative.")

    subtraction_point = float(subtraction_point)
    if not np.isfinite(subtraction_point):
        raise ValueError("subtraction_point must be finite.")

    if n_subtractions > 0 and threshold <= subtraction_point <= cutoff:
        raise ValueError(
            "For this real-axis implementation, subtraction_point must lie "
            "outside the integration interval."
        )

    return int(n_subtractions), subtraction_point


# ---------------------------------------------------------------------------
# Subtraction polynomial
# ---------------------------------------------------------------------------

def subtraction_polynomial(x, coefficients, center=0.0):
    """Evaluate a polynomial in ascending powers of ``x-center``.

    ``coefficients = [a0, a1, a2]`` means

        P(x) = a0 + a1*(x-center) + a2*(x-center)**2.

    Coefficients may be real or complex.  Scalar input produces a scalar;
    array input produces a complex NumPy array.
    """
    x_array = _finite_real_array(x, "x")
    coefficients = np.asarray(coefficients, dtype=complex)

    if coefficients.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional sequence.")
    if np.any(~np.isfinite(coefficients.real)) or np.any(
        ~np.isfinite(coefficients.imag)
    ):
        raise ValueError("coefficients must be finite.")

    shifted = x_array - float(center)
    result = np.zeros(x_array.shape, dtype=complex)

    # Horner evaluation for coefficients stored in ascending order.
    for coefficient in coefficients[::-1]:
        result = result * shifted + coefficient

    if x_array.ndim == 0:
        return complex(result)
    return result


# ---------------------------------------------------------------------------
# Interpolation helpers for precomputed Omega and hat functions
# ---------------------------------------------------------------------------

def make_interpolator(x_grid, values, kind="cubic", extrapolate=False):
    """Build a real-argument interpolator for real or complex samples.

    Parameters
    ----------
    x_grid : array_like
        Strictly increasing real grid.
    values : array_like
        Real or complex values on ``x_grid``.
    kind : {"linear", "cubic"}
        Interpolation method.
    extrapolate : bool
        If False, evaluating outside the grid raises ValueError.
    """
    x_grid = _finite_real_array(x_grid, "x_grid")
    values = np.asarray(values, dtype=complex)

    if x_grid.ndim != 1 or values.ndim != 1:
        raise ValueError("x_grid and values must be one-dimensional.")
    if x_grid.size != values.size:
        raise ValueError("x_grid and values must have the same length.")
    if x_grid.size < 2:
        raise ValueError("At least two interpolation points are required.")
    if np.any(np.diff(x_grid) <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")
    if np.any(~np.isfinite(values.real)) or np.any(~np.isfinite(values.imag)):
        raise ValueError("values must be finite.")

    if kind == "linear":
        raw_interpolator = interp1d(
            x_grid,
            values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
    elif kind == "cubic":
        raw_interpolator = CubicSpline(
            x_grid,
            values,
            extrapolate=True,
        )
    else:
        raise ValueError("kind must be either 'linear' or 'cubic'.")

    x_min = float(x_grid[0])
    x_max = float(x_grid[-1])

    def interpolator(x):
        x_array = _finite_real_array(x, "interpolation argument")
        if not extrapolate and np.any((x_array < x_min) | (x_array > x_max)):
            raise ValueError(
                f"Interpolation argument lies outside [{x_min}, {x_max}]."
            )

        result = np.asarray(raw_interpolator(x_array), dtype=complex)
        if x_array.ndim == 0:
            return complex(result)
        return result

    return interpolator


def interpolate_nested_hat_functions(
    x_grid,
    hat_values,
    kind="cubic",
    extrapolate=False,
):
    """Convert a nested hat-value dictionary into callable interpolators.

    For example, the output of ``angular_averages.hat_s`` on an array has the
    form::

        {"0": {"S0": values},
         "1": {"P0": values, "Pperp": values}}

    This function preserves the dictionary structure and replaces every
    value array by an interpolating function.
    """
    if not isinstance(hat_values, Mapping):
        raise TypeError("hat_values must be a nested mapping.")

    result = {}
    for channel, waves in hat_values.items():
        if not isinstance(waves, Mapping):
            raise TypeError("Each channel entry in hat_values must be a mapping.")
        result[channel] = {
            wave: make_interpolator(
                x_grid,
                values,
                kind=kind,
                extrapolate=extrapolate,
            )
            for wave, values in waves.items()
        }

    return result


# ---------------------------------------------------------------------------
# Dispersive integral
# ---------------------------------------------------------------------------

def _driving_function(
    xp,
    phase_shift,
    omega,
    hat_function,
    n_subtractions,
    subtraction_point,
    omega_floor,
):
    phase = _scalar_real(phase_shift, xp, "phase_shift")
    omega_value = _scalar_complex(omega, xp, "omega")
    omega_modulus = abs(omega_value)

    if omega_modulus <= omega_floor:
        raise ZeroDivisionError(
            f"|omega({xp})|={omega_modulus} is too small for the KT kernel."
        )

    hat_value = _scalar_complex(hat_function, xp, "hat_function")
    subtraction_denominator = (xp - subtraction_point) ** n_subtractions

    return (
        np.sin(phase)
        * hat_value
        / (subtraction_denominator * omega_modulus)
    )


def _complex_quad_regular(function, lower, upper, epsabs, epsrel, limit):
    real_value, real_error = quad(
        lambda xp: float(np.real(function(xp))),
        lower,
        upper,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    imag_value, imag_error = quad(
        lambda xp: float(np.imag(function(xp))),
        lower,
        upper,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return real_value + 1j * imag_value, float(np.hypot(real_error, imag_error))


def _complex_quad_cauchy(function, lower, upper, pole, epsabs, epsrel, limit):
    real_value, real_error = quad(
        lambda xp: float(np.real(function(xp))),
        lower,
        upper,
        weight="cauchy",
        wvar=pole,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    imag_value, imag_error = quad(
        lambda xp: float(np.imag(function(xp))),
        lower,
        upper,
        weight="cauchy",
        wvar=pole,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return real_value + 1j * imag_value, float(np.hypot(real_error, imag_error))


def _dispersive_integral_scalar(
    x,
    phase_shift,
    omega,
    hat_function,
    threshold,
    cutoff,
    n_subtractions,
    subtraction_point,
    epsabs,
    epsrel,
    integration_limit,
    omega_floor,
    boundary,
):
    endpoint_tolerance = 1.0e-13 * max(1.0, abs(threshold), abs(cutoff))
    if abs(x - threshold) <= endpoint_tolerance or abs(x - cutoff) <= endpoint_tolerance:
        raise ValueError(
            "The external point x must not equal threshold or cutoff. "
            "Use threshold + epsilon or cutoff - epsilon."
        )

    if n_subtractions > 0 and x == subtraction_point:
        return 0.0j, 0.0

    driving = lambda xp: _driving_function(
        xp=xp,
        phase_shift=phase_shift,
        omega=omega,
        hat_function=hat_function,
        n_subtractions=n_subtractions,
        subtraction_point=subtraction_point,
        omega_floor=omega_floor,
    )

    if threshold < x < cutoff:
        integral, error = _complex_quad_cauchy(
            driving,
            threshold,
            cutoff,
            x,
            epsabs,
            epsrel,
            integration_limit,
        )
        if boundary != "principal_value":
            residue_sign = 1.0 if boundary == "upper" else -1.0
            integral = integral + residue_sign * 1j * np.pi * driving(x)
    else:
        integral, error = _complex_quad_regular(
            lambda xp: driving(xp) / (xp - x),
            threshold,
            cutoff,
            epsabs,
            epsrel,
            integration_limit,
        )

    prefactor = (x - subtraction_point) ** n_subtractions / np.pi
    return prefactor * integral, abs(prefactor) * error / np.pi


def dispersive_integral(
    x,
    phase_shift,
    omega,
    hat_function,
    threshold,
    cutoff,
    n_subtractions=1,
    subtraction_point=0.0,
    epsabs=1.0e-9,
    epsrel=1.0e-8,
    integration_limit=200,
    omega_floor=1.0e-14,
    boundary="upper",
    return_error=False,
):
    """Evaluate the finite-cutoff KT dispersive integral.

    If x lies strictly between threshold and cutoff, the real-axis principal
    value is evaluated with Cauchy-weight quadrature.  Complex hat functions
    are integrated by treating their real and imaginary parts separately.

    ``boundary`` specifies the real-axis prescription when x is on the cut:

    * ``"upper"`` uses x+i0 and adds the positive imaginary residue;
    * ``"lower"`` uses x-i0 and adds the negative imaginary residue;
    * ``"principal_value"`` returns the principal-value part alone.
    """
    threshold, cutoff = _validate_interval(threshold, cutoff)
    n_subtractions, subtraction_point = _validate_subtractions(
        n_subtractions,
        subtraction_point,
        threshold,
        cutoff,
    )

    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("epsabs and epsrel must be positive.")
    if not isinstance(integration_limit, (int, np.integer)) or integration_limit < 1:
        raise ValueError("integration_limit must be a positive integer.")
    if omega_floor <= 0.0:
        raise ValueError("omega_floor must be positive.")
    if boundary not in {"upper", "lower", "principal_value"}:
        raise ValueError(
            "boundary must be 'upper', 'lower', or 'principal_value'."
        )

    x_array = _finite_real_array(x, "x")
    values = np.empty(x_array.shape, dtype=complex)
    errors = np.empty(x_array.shape, dtype=float)

    if x_array.ndim == 0:
        value, error = _dispersive_integral_scalar(
            float(x_array),
            phase_shift,
            omega,
            hat_function,
            threshold,
            cutoff,
            n_subtractions,
            subtraction_point,
            epsabs,
            epsrel,
            int(integration_limit),
            omega_floor,
            boundary,
        )
        return (value, error) if return_error else value

    for index in np.ndindex(x_array.shape):
        values[index], errors[index] = _dispersive_integral_scalar(
            float(x_array[index]),
            phase_shift,
            omega,
            hat_function,
            threshold,
            cutoff,
            n_subtractions,
            subtraction_point,
            epsabs,
            epsrel,
            int(integration_limit),
            omega_floor,
            boundary,
        )

    return (values, errors) if return_error else values


# ---------------------------------------------------------------------------
# One KT update
# ---------------------------------------------------------------------------

def kt_update(
    x,
    phase_shift,
    omega,
    hat_function,
    threshold,
    cutoff,
    subtraction_coefficients,
    n_subtractions=None,
    subtraction_point=0.0,
    polynomial_center=0.0,
    tail_correction=0.0,
    epsabs=1.0e-9,
    epsrel=1.0e-8,
    integration_limit=200,
    omega_floor=1.0e-14,
    boundary="upper",
    return_parts=False,
):
    """Perform one channel-independent Khuri-Treiman update.

    ``tail_correction`` is added inside the Omnes brackets.  It can be a
    constant or a callable of x.  Set it to zero when the region above the
    finite cutoff is deliberately neglected.
    """
    coefficients = np.asarray(subtraction_coefficients, dtype=complex)
    if coefficients.ndim != 1:
        raise ValueError("subtraction_coefficients must be one-dimensional.")

    if n_subtractions is None:
        n_subtractions = int(coefficients.size)

    if not isinstance(n_subtractions, (int, np.integer)):
        raise TypeError("n_subtractions must be an integer.")
    if n_subtractions < 0:
        raise ValueError("n_subtractions must be non-negative.")
    if coefficients.size > n_subtractions:
        raise ValueError(
            "A representation with n subtractions can contain at most n "
            "subtraction-polynomial coefficients."
        )

    x_array = _finite_real_array(x, "x")
    polynomial = subtraction_polynomial(
        x_array,
        coefficients,
        center=polynomial_center,
    )
    integral = dispersive_integral(
        x=x_array,
        phase_shift=phase_shift,
        omega=omega,
        hat_function=hat_function,
        threshold=threshold,
        cutoff=cutoff,
        n_subtractions=int(n_subtractions),
        subtraction_point=subtraction_point,
        epsabs=epsabs,
        epsrel=epsrel,
        integration_limit=integration_limit,
        omega_floor=omega_floor,
        boundary=boundary,
    )

    if x_array.ndim == 0:
        omega_external = _scalar_complex(omega, float(x_array), "omega")
        tail = _scalar_complex(tail_correction, float(x_array), "tail_correction")
        bracket = polynomial + integral + tail
        updated = omega_external * bracket
    else:
        omega_external = np.empty(x_array.shape, dtype=complex)
        tail = np.empty(x_array.shape, dtype=complex)
        for index in np.ndindex(x_array.shape):
            omega_external[index] = _scalar_complex(
                omega,
                float(x_array[index]),
                "omega",
            )
            tail[index] = _scalar_complex(
                tail_correction,
                float(x_array[index]),
                "tail_correction",
            )
        bracket = polynomial + integral + tail
        updated = omega_external * bracket

    if return_parts:
        return {
            "updated": updated,
            "omega": omega_external,
            "polynomial": polynomial,
            "integral": integral,
            "tail": tail,
            "bracket": bracket,
        }
    return updated


# ---------------------------------------------------------------------------
# Numerical self-checks
# ---------------------------------------------------------------------------

def run_self_checks(atol=2.0e-9):
    """Check the polynomial, interpolation and principal-value conventions."""
    errors = {}

    polynomial_value = subtraction_polynomial(
        1.5,
        [1.0, 2.0, 3.0],
        center=0.5,
    )
    errors["polynomial"] = abs(polynomial_value - 6.0)

    grid = np.linspace(0.0, 2.0, 9)
    samples = (1.0 + 0.5j) + (2.0 - 0.25j) * grid
    interpolator = make_interpolator(grid, samples, kind="cubic")
    interpolation_point = 0.73
    expected_interpolation = (
        (1.0 + 0.5j) + (2.0 - 0.25j) * interpolation_point
    )
    errors["interpolation"] = abs(
        interpolator(interpolation_point) - expected_interpolation
    )

    threshold = 1.0
    cutoff = 3.0
    constant_hat = 1.0 + 0.25j
    phase_shift = lambda xp: np.pi / 2.0
    omega = lambda xp: 1.0 + 0.0j
    hat_function = lambda xp: constant_hat

    for label, x in (("regular_integral", 0.5), ("principal_value", 2.0)):
        numerical = dispersive_integral(
            x=x,
            phase_shift=phase_shift,
            omega=omega,
            hat_function=hat_function,
            threshold=threshold,
            cutoff=cutoff,
            n_subtractions=1,
            subtraction_point=0.0,
            epsabs=1.0e-11,
            epsrel=1.0e-11,
            boundary="principal_value",
        )
        logarithm = (
            np.log(abs((cutoff - x) / (threshold - x)))
            - np.log(cutoff / threshold)
        )
        analytic = constant_hat * logarithm / np.pi
        errors[label] = abs(numerical - analytic)

    upper_rim = dispersive_integral(
        x=2.0,
        phase_shift=phase_shift,
        omega=omega,
        hat_function=hat_function,
        threshold=threshold,
        cutoff=cutoff,
        n_subtractions=1,
        subtraction_point=0.0,
        epsabs=1.0e-11,
        epsrel=1.0e-11,
        boundary="upper",
    )
    principal_value = constant_hat * (-np.log(3.0)) / np.pi
    errors["upper_rim"] = abs(
        upper_rim - (principal_value + 1j * constant_hat)
    )

    subtraction_value = kt_update(
        x=0.0,
        phase_shift=phase_shift,
        omega=omega,
        hat_function=hat_function,
        threshold=threshold,
        cutoff=cutoff,
        subtraction_coefficients=[2.0],
        n_subtractions=1,
        subtraction_point=0.0,
    )
    errors["subtraction_point"] = abs(subtraction_value - 2.0)

    if max(errors.values()) > atol:
        raise RuntimeError(f"kt_equations self-check failed: {errors}")

    return errors


__all__ = [
    "subtraction_polynomial",
    "make_interpolator",
    "interpolate_nested_hat_functions",
    "dispersive_integral",
    "kt_update",
    "run_self_checks",
]
