# ============================================================
# Cell 1: Physical parameters
# ============================================================

# 质量单位统一为 GeV
m_D1 = 2.420000
m_D  = 1.867000
m_pi = 0.13957039


# 检查质量是否为正
if m_D1 <= 0 or m_D <= 0 or m_pi <= 0:
    raise ValueError("所有粒子质量必须为正数。")


# 检查三体衰变是否运动学允许
if m_D1 <= m_D + 2.0*m_pi:
    raise ValueError(
        "D1 -> D pi pi 在当前质量下不允许。"
    )


# Mandelstam identity:
# s + t + u = Sigma
Sigma = (
    m_D1**2
    + m_D**2
    + 2.0*m_pi**2
)


# s-channel physical range
s_decay_min = (2.0*m_pi)**2
s_decay_max = (m_D1 - m_D)**2


# t-channel global physical range
t_decay_min = (m_D + m_pi)**2
t_decay_max = (m_D1 - m_pi)**2


# u-channel global physical range
u_decay_min = t_decay_min
u_decay_max = t_decay_max


print(f"m_D1  = {m_D1:.9f} GeV")
print(f"m_D   = {m_D:.9f} GeV")
print(f"m_pi  = {m_pi:.9f} GeV")
print(f"Sigma = {Sigma:.9f} GeV^2")

print(
    f"s range = "
    f"[{s_decay_min:.9f}, "
    f"{s_decay_max:.9f}] GeV^2"
)

import numpy as np


# ============================================================
# Källén 函数
# ============================================================

def kallen(x, y, z):
    """
    Källén 函数：

    lambda(x,y,z)
    = x^2 + y^2 + z^2 - 2xy - 2xz - 2yz
    """
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
    对物理区内应当非负的量计算平方根。

    阈值附近可能因为浮点误差出现很小的负数，
    例如 -1e-16，此时将其截断为 0。

    如果出现明显负数，则说明输入可能超出物理区。
    """
    x = np.asarray(x, dtype=float)

    if np.any(x < -tolerance):
        raise ValueError(
            "平方根内部出现明显负数，"
            "输入可能位于当前函数所定义的物理区之外。"
        )

    return np.sqrt(np.clip(x, 0.0, None))

# ============================================================
# s-channel kinematics
#
# s = (p_pi1 + p_pi2)^2
# ============================================================

def sigma_pi(s):
    """
    pi-pi 两体相空间因子：

    sigma_pi(s) = sqrt(1 - 4*m_pi^2/s)
    """
    s = np.asarray(s, dtype=float)

    if np.any(s <= 0.0):
        raise ValueError("s 必须大于 0。")

    return physical_sqrt(
        1.0 - 4.0*m_pi**2/s
    )


def lambda_D1_D_s(s):
    """
    lambda(m_D1^2, m_D^2, s)
    """
    return kallen(
        m_D1**2,
        m_D**2,
        s
    )

def E_pi_s(s):
    """
    pi-pi 质心系中，每个 pion 的能量。
    """
    return np.sqrt(s) / 2.0


def q_pi_s(s):
    """
    pi-pi 质心系中，每个 pion 的三动量大小。
    """
    return 0.5 * physical_sqrt(
        s - 4.0*m_pi**2
    )


def E_D_s(s):
    """
    pi-pi 质心系中，D 介子的能量。
    """
    return (
        m_D1**2
        - m_D**2
        - s
    ) / (2.0*np.sqrt(s))


def p_D_s(s):
    """
    pi-pi 质心系中，D 介子的三动量大小。
    """
    return (
        physical_sqrt(lambda_D1_D_s(s))
        / (2.0*np.sqrt(s))
    )

# ============================================================
# s-channel Mandelstam-variable mapping
#
# Convention:
#
#             u - t
# z_s = -------------------------------
#        sqrt(lambda_1)*sqrt(lambda_2)/s
# ============================================================

def t_of_s_z(s, z_s):
    """
    给定 s 和 z_s，计算 t。
    """
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
    给定 s 和 z_s，计算 u。
    """
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


def z_s_from_s_t(s, t):
    """
    从 s 和 t 反求 z_s。
    """
    u = Sigma - s - t

    denominator = (
        physical_sqrt(
            kallen(s, m_D1**2, m_D**2)
        )
        * physical_sqrt(
            kallen(s, m_pi**2, m_pi**2)
        )
        / s
    )

    if np.any(np.abs(denominator) < 1e-14):
        raise ValueError(
            "分母接近零，当前 s 位于运动学端点。"
        )

    return (u - t) / denominator

# ============================================================
# Cell 5: Kinematic limits at fixed s
# ============================================================

def t_limits(s):
    """
    固定 s 时，返回 t 的允许范围：

    t_min(s) <= t <= t_max(s)

    当前角变量约定：
        z_s = (u-t)/(...)
    """
    t_min = t_of_s_z(s, 1.0)
    t_max = t_of_s_z(s, -1.0)

    return t_min, t_max


def u_limits(s):
    """
    固定 s 时，返回 u 的允许范围：

    u_min(s) <= u <= u_max(s)
    """
    u_min = u_of_s_z(s, -1.0)
    u_max = u_of_s_z(s, 1.0)

    return u_min, u_max

# ============================================================
# Cell 6: Test the s-channel kinematics
# ============================================================

# 选择 s 物理区内部的一个测试点
s_test = 0.5 * (
    s_decay_min + s_decay_max
)

# 选择若干 z_s 测试点
z_test = np.array([
    -1.0,
    -0.5,
     0.0,
     0.5,
     1.0
])

# 根据 s 和 z_s 计算 t、u
t_test = t_of_s_z(s_test, z_test)
u_test = u_of_s_z(s_test, z_test)


# ============================================================
# Test 1: Mandelstam identity
#
# s + t + u = Sigma
# ============================================================

assert np.allclose(
    s_test + t_test + u_test,
    Sigma
)


# ============================================================
# Test 2: Verify the definition of z_s
#
#             u - t
# z_s = -------------------------------
#        sqrt(lambda_1)*sqrt(lambda_2)/s
# ============================================================

denominator_test = (
    physical_sqrt(
        kallen(s_test, m_D1**2, m_D**2)
    )
    * physical_sqrt(
        kallen(s_test, m_pi**2, m_pi**2)
    )
    / s_test
)

z_from_definition = (
    u_test - t_test
) / denominator_test

assert np.allclose(
    z_from_definition,
    z_test
)


# ============================================================
# Test 3: Pion exchange
#
# t <-> u corresponds to z_s -> -z_s
# ============================================================

assert np.allclose(
    t_of_s_z(s_test, z_test),
    u_of_s_z(s_test, -z_test)
)


# ============================================================
# Test 4: Check the kinematic limits
# ============================================================

t_min_test, t_max_test = t_limits(s_test)
u_min_test, u_max_test = u_limits(s_test)

assert np.all(t_test >= t_min_test)
assert np.all(t_test <= t_max_test)

assert np.all(u_test >= u_min_test)
assert np.all(u_test <= u_max_test)


# ============================================================
# Display results
# ============================================================

print("All s-channel kinematic tests passed.")

print()
print(f"s = {s_test:.9f} GeV^2")

print()
print(
    f"t range = "
    f"[{t_min_test:.9f}, {t_max_test:.9f}] GeV^2"
)

print(
    f"u range = "
    f"[{u_min_test:.9f}, {u_max_test:.9f}] GeV^2"
)

print()
print("z_s values:")
print(z_test)

print()
print("t values:")
print(t_test)

print()
print("u values:")
print(u_test)

print()
print("z_s reconstructed directly from its definition:")
print(z_from_definition)
