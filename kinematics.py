"""
All masses are given in GeV.
All Mandelstam variables are in GeV^2.

The functions in this file use real square roots and are intended
for calculations in the real decay region. Complex analytic
continuation will require a separate square-root prescription.
"""

import numpy as np

# 1. Physical parameters
m_D1 = 2.420000       # GeV
m_D = 1.867000        # GeV
m_pi = 0.13957039     # GeV
Sigma = (m_D1**2 + m_D**2 + 2.0*m_pi**2)
subtraction_center = Sigma / 3.0


Q_value = (m_D1 - m_D - 2.0*m_pi)
# Mass combination appearing in z_t and z_u.
Delta_Dpi = ( (m_D1**2 - m_pi**2) * (m_D**2 - m_pi**2) )

s_threshold = (2.0*m_pi)**2
s_decay_min = s_threshold
s_decay_max = (m_D1 - m_D)**2
# t = (p_D + p_pi1)^2
t_threshold = (m_D + m_pi)**2
t_decay_min = t_threshold
t_decay_max = (m_D1 - m_pi)**2
# u = (p_D + p_pi2)^2
u_threshold = t_threshold
u_decay_min = t_decay_min
u_decay_max = t_decay_max
def kallen(x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    return (
        x**2
        + y**2
        + z**2
        - 2.0*x*y
        - 2.0*x*z
        - 2.0*y*z
    )


def physical_sqrt(x, tolerance=1e-12):
    """
    Square root for calculations in the real physical region.

    Very small negative values caused by floating-point errors
    are replaced by zero.

    A significantly negative value raises an error.

    This function must not be used for complex analytic
    continuation.
    """
    x = np.asarray(x, dtype=float)

    if np.any(x < -tolerance):
        raise ValueError(
            "The square-root argument is negative. "
            "The input may lie outside the real physical region."
        )

    return np.sqrt(
        np.clip(x, 0.0, None)
    )


# ============================================================
# 5. Useful s-channel quantities
# ============================================================

def sigma_pi(s):
    """
    Two-pion phase-space factor:

    sigma_pi(s) = sqrt(1 - 4*m_pi^2/s)
    """
    s = np.asarray(s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    return physical_sqrt(
        1.0 - 4.0*m_pi**2/s
    )


def lambda_D1_D_s(s):
    """
    Return

    lambda(s, m_D1^2, m_D^2).
    """
    return kallen(
        s,
        m_D1**2,
        m_D**2
    )


def E_pi_s(s):
    """
    Pion energy in the pi-pi center-of-mass frame:

    E_pi(s) = sqrt(s)/2.
    """
    s = np.asarray(s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    return np.sqrt(s) / 2.0


def q_pi_s(s):
    """
    Pion momentum in the pi-pi center-of-mass frame:

               sqrt(lambda(s,m_pi^2,m_pi^2))
    q_pi(s) = --------------------------------
                         2*sqrt(s)
    """
    s = np.asarray(s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    return (
        physical_sqrt(
            kallen(s, m_pi**2, m_pi**2)
        )
        / (2.0*np.sqrt(s))
    )


def E_D_s(s):
    """
    D-meson energy in the pi-pi center-of-mass frame:

             m_D1^2 - m_D^2 - s
    E_D(s) = --------------------
                    2*sqrt(s)
    """
    s = np.asarray(s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    return (
        m_D1**2
        - m_D**2
        - s
    ) / (2.0*np.sqrt(s))


def p_D_s(s):
    """
    D-meson momentum in the pi-pi center-of-mass frame:

              sqrt(lambda(s,m_D1^2,m_D^2))
    p_D(s) = --------------------------------
                        2*sqrt(s)
    """
    s = np.asarray(s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    return (
        physical_sqrt(
            kallen(s, m_D1**2, m_D**2)
        )
        / (2.0*np.sqrt(s))
    )


# ============================================================
# 6. s-channel Mandelstam-variable mapping
# ============================================================

def t_of_s_z(s, z_s):
    """
    Calculate t from s and z_s.

    Angle convention:

                  u - t
    z_s = ----------------------------------
          sqrt(lambda_s1)*sqrt(lambda_s2)/s

    Therefore:

        t = (Sigma - s - denominator*z_s)/2.
    """
    s = np.asarray(s, dtype=float)
    z_s = np.asarray(z_s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    denominator = (
        physical_sqrt(
            kallen(s, m_D1**2, m_D**2)
        )
        * physical_sqrt(
            kallen(s, m_pi**2, m_pi**2)
        )
        / s
    )

    return (
        Sigma
        - s
        - denominator*z_s
    ) / 2.0


def u_of_s_z(s, z_s):
    """
    Calculate u from s and z_s.

    Angle convention:

                  u - t
    z_s = ----------------------------------
          sqrt(lambda_s1)*sqrt(lambda_s2)/s

    Therefore:

        u = (Sigma - s + denominator*z_s)/2.
    """
    s = np.asarray(s, dtype=float)
    z_s = np.asarray(z_s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s must be positive.")

    denominator = (
        physical_sqrt(
            kallen(s, m_D1**2, m_D**2)
        )
        * physical_sqrt(
            kallen(s, m_pi**2, m_pi**2)
        )
        / s
    )

    return (
        Sigma
        - s
        + denominator*z_s
    ) / 2.0


# ============================================================
# 7. Kinematic limits at fixed s
# ============================================================

def t_limits(s):
    """
    Return the allowed t interval at fixed s:

    t_min(s) <= t <= t_max(s).

    With the current convention:

    z_s = +1 gives t_min,
    z_s = -1 gives t_max.
    """
    t_min = t_of_s_z(s, 1.0)
    t_max = t_of_s_z(s, -1.0)

    return t_min, t_max


def u_limits(s):
    """
    Return the allowed u interval at fixed s:

    u_min(s) <= u <= u_max(s).

    With the current convention:

    z_s = -1 gives u_min,
    z_s = +1 gives u_max.
    """
    u_min = u_of_s_z(s, -1.0)
    u_max = u_of_s_z(s, 1.0)

    return u_min, u_max


# ============================================================
# 8. t-channel Mandelstam-variable mapping
# ============================================================

def s_of_t_z(t, z_t):
    """
    Calculate s from t and z_t.

    Angle convention:

             s - u + Delta_Dpi/t
    z_t = ----------------------------------
          sqrt(lambda_t1)*sqrt(lambda_t2)/t

    Therefore:

        s = (
            Sigma - t
            + denominator*z_t
            - Delta_Dpi/t
        ) / 2.
    """
    t = np.asarray(t, dtype=float)
    z_t = np.asarray(z_t, dtype=float)

    if np.any(t <= 0.0):
        raise ValueError("t must be positive.")

    denominator = (
        physical_sqrt(
            kallen(t, m_D1**2, m_pi**2)
        )
        * physical_sqrt(
            kallen(t, m_D**2, m_pi**2)
        )
        / t
    )

    return (
        Sigma
        - t
        + denominator*z_t
        - Delta_Dpi/t
    ) / 2.0


def u_of_t_z(t, z_t):
    """
    Calculate u from t and z_t.

    Angle convention:

             s - u + Delta_Dpi/t
    z_t = ----------------------------------
          sqrt(lambda_t1)*sqrt(lambda_t2)/t

    Therefore:

        u = (
            Sigma - t
            - denominator*z_t
            + Delta_Dpi/t
        ) / 2.
    """
    t = np.asarray(t, dtype=float)
    z_t = np.asarray(z_t, dtype=float)

    if np.any(t <= 0.0):
        raise ValueError("t must be positive.")

    denominator = (
        physical_sqrt(
            kallen(t, m_D1**2, m_pi**2)
        )
        * physical_sqrt(
            kallen(t, m_D**2, m_pi**2)
        )
        / t
    )

    return (
        Sigma
        - t
        - denominator*z_t
        + Delta_Dpi/t
    ) / 2.0


# ============================================================
# 9. Kinematic limits at fixed t
# ============================================================

def s_limits_at_t(t):
    """
    Return the allowed s interval at fixed t.
    """
    s_min = s_of_t_z(t, -1.0)
    s_max = s_of_t_z(t, 1.0)

    return s_min, s_max


def u_limits_at_t(t):
    """
    Return the allowed u interval at fixed t.
    """
    u_min = u_of_t_z(t, 1.0)
    u_max = u_of_t_z(t, -1.0)

    return u_min, u_max


# ============================================================
# 10. u-channel Mandelstam-variable mapping
# ============================================================

def s_of_u_z(u, z_u):
    """
    Calculate s from u and z_u.

    Angle convention:

             s - t + Delta_Dpi/u
    z_u = ----------------------------------
          sqrt(lambda_u1)*sqrt(lambda_u2)/u

    Therefore:

        s = (
            Sigma - u
            + denominator*z_u
            - Delta_Dpi/u
        ) / 2.
    """
    u = np.asarray(u, dtype=float)
    z_u = np.asarray(z_u, dtype=float)

    if np.any(u <= 0.0):
        raise ValueError("u must be positive.")

    denominator = (
        physical_sqrt(
            kallen(u, m_D1**2, m_pi**2)
        )
        * physical_sqrt(
            kallen(u, m_D**2, m_pi**2)
        )
        / u
    )

    return (
        Sigma
        - u
        + denominator*z_u
        - Delta_Dpi/u
    ) / 2.0


def t_of_u_z(u, z_u):
    """
    Calculate t from u and z_u.

    Angle convention:

             s - t + Delta_Dpi/u
    z_u = ----------------------------------
          sqrt(lambda_u1)*sqrt(lambda_u2)/u

    Therefore:

        t = (
            Sigma - u
            - denominator*z_u
            + Delta_Dpi/u
        ) / 2.
    """
    u = np.asarray(u, dtype=float)
    z_u = np.asarray(z_u, dtype=float)

    if np.any(u <= 0.0):
        raise ValueError("u must be positive.")

    denominator = (
        physical_sqrt(
            kallen(u, m_D1**2, m_pi**2)
        )
        * physical_sqrt(
            kallen(u, m_D**2, m_pi**2)
        )
        / u
    )

    return (
        Sigma
        - u
        - denominator*z_u
        + Delta_Dpi/u
    ) / 2.0


# ============================================================
# 11. Kinematic limits at fixed u
# ============================================================

def s_limits_at_u(u):
    """
    Return the allowed s interval at fixed u.
    """
    s_min = s_of_u_z(u, -1.0)
    s_max = s_of_u_z(u, 1.0)

    return s_min, s_max


def t_limits_at_u(u):
    """
    Return the allowed t interval at fixed u.
    """
    t_min = t_of_u_z(u, 1.0)
    t_max = t_of_u_z(u, -1.0)

    return t_min, t_max