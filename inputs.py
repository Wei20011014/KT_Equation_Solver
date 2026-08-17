# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
#This is some inputs#
# ============================================================
# 1. 物理参数输入
# ============================================================

# 统一采用 GeV 作为质量单位
# 以下数值暂时只是示例，正式计算时再替换

m_D1 = 2.420000       # D1 质量，GeV
m_D  = 1.867000       # D 质量，GeV
m_pi = 0.13957039     # pion 质量，GeV


# ============================================================
# 2. 检查输入是否合法
# ============================================================

# 所有质量必须为正数
if m_D1 <= 0 or m_D <= 0 or m_pi <= 0:
    raise ValueError("所有粒子质量都必须为正数。")

# 检查 D1 -> D + pi + pi 是否运动学允许
if m_D1 <= m_D + 2.0 * m_pi:
    raise ValueError(
        "D1 -> D pi pi 在当前质量输入下运动学不允许："
        "需要满足 m_D1 > m_D + 2*m_pi。"
    )


# ============================================================
# 3. 计算常用的派生物理量
# ============================================================

# Mandelstam 变量满足：
# s + t + u = Sigma
Sigma = m_D1**2 + m_D**2 + 2.0 * m_pi**2

# 中心化变量的展开点：
# s_bar = s - subtraction_center
subtraction_center = Sigma / 3.0

# 衰变释放的能量
Q_value = m_D1 - m_D - 2.0 * m_pi


# ============================================================
# 4. s = (p_pi1 + p_pi2)^2 的范围
# ============================================================

# pi-pi 右手割线阈值
s_threshold = (2.0 * m_pi)**2

# Dalitz plot 中 s 的全局范围
s_decay_min = s_threshold
s_decay_max = (m_D1 - m_D)**2


# ============================================================
# 5. t = (p_D + p_pi1)^2 的范围
# ============================================================

# D-pi 右手割线阈值
t_threshold = (m_D + m_pi)**2

# Dalitz plot 中 t 的全局范围
t_decay_min = t_threshold
t_decay_max = (m_D1 - m_pi)**2


# ============================================================
# 6. u = (p_D + p_pi2)^2 的范围
# ============================================================

# 两个 pion 质量相同，因此 u 道与 t 道阈值相同
u_threshold = t_threshold

u_decay_min = t_decay_min
u_decay_max = t_decay_max


# ============================================================
# 7. 显示计算结果
# ============================================================

print("Physical parameters")
print("-------------------")
print(f"m_D1 = {m_D1:.9f} GeV")
print(f"m_D  = {m_D:.9f} GeV")
print(f"m_pi = {m_pi:.9f} GeV")

print()
print(f"Q value = {Q_value:.9f} GeV")
print(f"Sigma   = {Sigma:.9f} GeV^2")
print(f"Sigma/3 = {subtraction_center:.9f} GeV^2")

print()
print(
    f"s physical range: "
    f"[{s_decay_min:.9f}, {s_decay_max:.9f}] GeV^2"
)

print(
    f"t physical range: "
    f"[{t_decay_min:.9f}, {t_decay_max:.9f}] GeV^2"
)

print(
    f"u physical range: "
    f"[{u_decay_min:.9f}, {u_decay_max:.9f}] GeV^2"
)