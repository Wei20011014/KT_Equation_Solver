"""
Numerical Omnes function
========================

This module evaluates the once-subtracted Omnes function

                 x
Omega(x) = exp[ ----- integral dx'
                pi

                 delta(x')
              ---------------- ]
              x' (x' - x - i0)

with the normalization

    Omega(0) = 1.

Conventions
-----------
1. x, threshold and cutoff are in GeV^2.
2. The phase-shift function returns radians.
3. The result on the right-hand cut is evaluated on the
   upper rim:

       Omega(x+i0) = |Omega(x)| exp[i delta(x)].

4. The phase shift is supplied numerically only up to cutoff.
   An optional constant high-energy tail can be included.

This module contains no channel-specific phase shifts,
physical masses or plotting code.
"""

import numpy as np

from scipy.integrate import quad


__all__ = [
    "omnes",
]


# ============================================================
# 1. Evaluate scalar phase-shift values
# ============================================================

def _phase_value(
    phase_shift,
    x
):
    """
    Evaluate a phase-shift function at one scalar point.

    Parameters
    ----------
    phase_shift:
        Callable phase_shift(x).

    x:
        Scalar invariant in GeV^2.

    Returns
    -------
    Scalar phase shift in radians.
    """
    value = phase_shift(
        float(x)
    )

    value = np.asarray(value)

    if value.size != 1:
        raise ValueError(
            "The phase-shift function must return exactly "
            "one value for scalar input."
        )

    value = float(
        value.item()
    )

    if not np.isfinite(value):
        raise ValueError(
            f"The phase shift is not finite at x={x:.16g}."
        )

    return value


# ============================================================
# 2. Scalar Omnes evaluation
# ============================================================

def _omnes_scalar(
    x,
    phase_shift,
    threshold,
    cutoff,
    tail_phase=None,
    epsabs=1e-9,
    epsrel=1e-9,
    integration_limit=300,
):
    """
    Evaluate Omega(x+i0) for one real x.

    Parameters
    ----------
    x:
        Real invariant in GeV^2.

    phase_shift:
        Function delta(x), returning radians.

    threshold:
        Right-hand-cut threshold in GeV^2.

    cutoff:
        Upper end of the numerical phase-shift input,
        in GeV^2.

    tail_phase:
        High-energy-tail prescription.

        If None:
            truncate the integral at cutoff.

        If a number:
            assume delta(x') = tail_phase for
            x' >= cutoff.

    epsabs, epsrel:
        Absolute and relative tolerances passed to
        scipy.integrate.quad.

    integration_limit:
        Maximum number of integration subintervals.

    Returns
    -------
    Complex value Omega(x+i0).
    """
    x = float(x)
    threshold = float(threshold)
    cutoff = float(cutoff)

    if not np.isfinite(x):
        raise ValueError(
            "x must be finite."
        )

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite."
        )

    if not np.isfinite(cutoff):
        raise ValueError(
            "cutoff must be finite."
        )

    if threshold <= 0.0:
        raise ValueError(
            "threshold must be positive."
        )

    if cutoff <= threshold:
        raise ValueError(
            "cutoff must be larger than threshold."
        )

    if x >= cutoff:
        raise ValueError(
            f"x={x:.16g} must be smaller than "
            f"cutoff={cutoff:.16g}."
        )

    if epsabs <= 0.0:
        raise ValueError(
            "epsabs must be positive."
        )

    if epsrel <= 0.0:
        raise ValueError(
            "epsrel must be positive."
        )

    if integration_limit <= 0:
        raise ValueError(
            "integration_limit must be positive."
        )

    # Exact normalization of the once-subtracted
    # Omnes function.
    if x == 0.0:
        return 1.0 + 0.0j

    # scipy's Cauchy-weight routine cannot place the pole
    # exactly at an integration endpoint.
    if x == threshold:
        raise ValueError(
            "Do not evaluate exactly at threshold. "
            "Use threshold plus or minus a small epsilon."
        )

    # ========================================================
    # Finite-range dispersive integral
    # ========================================================

    if threshold < x < cutoff:
        # On the right-hand cut:
        #
        # PV integral_threshold^cutoff
        # [delta(x')/x'] / (x'-x) dx'.
        principal_value, integration_error = quad(
            lambda xp: (
                _phase_value(
                    phase_shift,
                    xp
                )
                / xp
            ),
            threshold,
            cutoff,
            weight="cauchy",
            wvar=x,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=integration_limit,
        )

        log_modulus = (
            x / np.pi
        ) * principal_value

    else:
        # Below threshold, x'-x never vanishes and the
        # integral is an ordinary real integral.
        regular_integral, integration_error = quad(
            lambda xp: (
                _phase_value(
                    phase_shift,
                    xp
                )
                / (
                    xp
                    * (xp - x)
                )
            ),
            threshold,
            cutoff,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=integration_limit,
        )

        log_modulus = (
            x / np.pi
        ) * regular_integral

    if not np.isfinite(log_modulus):
        raise RuntimeError(
            f"The Omnes integral is not finite at "
            f"x={x:.16g}."
        )

    # ========================================================
    # Constant high-energy tail
    # ========================================================

    if tail_phase is not None:
        tail_phase = float(
            tail_phase
        )

        if not np.isfinite(tail_phase):
            raise ValueError(
                "tail_phase must be finite."
            )

        # If delta(x') = delta_inf above cutoff:
        #
        # x/pi * integral_cutoff^infinity
        # delta_inf/[x'(x'-x)] dx'
        #
        # = delta_inf/pi
        #   * log[cutoff/(cutoff-x)].
        log_modulus += (
            tail_phase / np.pi
        ) * np.log(
            cutoff
            / (cutoff - x)
        )

    # ========================================================
    # Phase on the upper rim of the right-hand cut
    # ========================================================

    if threshold < x < cutoff:
        phase_on_cut = _phase_value(
            phase_shift,
            x
        )
    else:
        phase_on_cut = 0.0

    return np.exp(
        log_modulus
        + 1j*phase_on_cut
    )


# ============================================================
# 3. Public scalar/array interface
# ============================================================

def omnes(
    x,
    phase_shift,
    threshold,
    cutoff,
    tail_phase=None,
    epsabs=1e-9,
    epsrel=1e-9,
    integration_limit=300,
):
    """
    Evaluate the once-subtracted Omnes function.

    Scalar or array input is supported.

    Parameters
    ----------
    x:
        Real scalar or array in GeV^2.

    phase_shift:
        Callable delta(x), returning radians.

    threshold:
        Right-hand-cut threshold in GeV^2.

    cutoff:
        Upper end of the available phase-shift range
        in GeV^2.

    tail_phase:
        None:
            truncate the integral at cutoff.

        Scalar:
            assume a constant phase above cutoff.

    epsabs, epsrel:
        Numerical integration tolerances.

    integration_limit:
        Maximum number of scipy integration subintervals.

    Returns
    -------
    If x is scalar:
        one complex number.

    If x is an array:
        a complex NumPy array with the same shape as x.
    """
    x_array = np.asarray(
        x,
        dtype=float
    )

    # Scalar input
    if x_array.ndim == 0:
        return _omnes_scalar(
            x=float(x_array),
            phase_shift=phase_shift,
            threshold=threshold,
            cutoff=cutoff,
            tail_phase=tail_phase,
            epsabs=epsabs,
            epsrel=epsrel,
            integration_limit=integration_limit,
        )

    # Array input
    result = np.empty(
        x_array.shape,
        dtype=complex
    )

    for index in np.ndindex(
        x_array.shape
    ):
        result[index] = _omnes_scalar(
            x=x_array[index],
            phase_shift=phase_shift,
            threshold=threshold,
            cutoff=cutoff,
            tail_phase=tail_phase,
            epsabs=epsabs,
            epsrel=epsrel,
            integration_limit=integration_limit,
        )

    return result