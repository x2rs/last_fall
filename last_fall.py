from scipy.stats import t
import numpy as np

# 重力加速度
g = 9.7940


class CalculationUnit:
    def __init__(self, name: str, data: list, B_uncertainty: float) -> None:
        self.delta_B = B_uncertainty
        self.name = name
        self.data = data
        self.mean = np.mean(self.data)
        if type(data) == list:
            self.n = len(data)
        else:
            self.n = 1

    # 标准差
    def sigma(self):
        if self.n == 1:
            return None
        return np.std(self.data, ddof=1)

    # A类不确定度
    def delta_A(self):
        if self.n == 1:
            return None
        return get_t_095_div_sqrt_n(self.n) * self.sigma()

    # 合成不确定度
    def uncertainty(self):
        if self.delta_A() is None:
            return self.delta_B
        return (self.delta_A() ** 2 + self.delta_B**2) ** 0.5

    # 相对不确定度(%)
    def relative_uncertainty(self):
        return self.uncertainty() / self.mean * 100

    # 表格
    def evaluate(self, part):
        return {
            "name": self.name,
            "mean": self.mean,
            "sigma": self.sigma(),
            "delta_A": self.delta_A(),
            "delta_B": self.delta_B,
            "u": self.uncertainty(),
            "u_r": self.relative_uncertainty(),
            "part_abs": np.abs(part),
            "part_times_u_abs": np.abs(part * self.uncertainty()),
        }

    # 真值
    def trueValue(self):
        return {
            "mean": self.mean,
            "u": self.uncertainty(),
        }


# 获取t_0.95/sqrt(n)
__T_095_DIV_SQRT_N = {
    3: 2.48,
    4: 1.59,
    5: 1.204,
    6: 1.05,
    7: 0.926,
    8: 0.834,
    9: 0.770,
    10: 0.715,
    15: 0.553,
    20: 0.467,
}


def get_t_095_div_sqrt_n(n: int):
    if n in __T_095_DIV_SQRT_N:
        return __T_095_DIV_SQRT_N[n]
    else:
        print(f"t_0.95/sqrt({n}) not found")


# 室温(Celsius)
T_temperature = 21.5


data_dict = {
    "t_fallTime": CalculationUnit(
        "t", [9.62, 9.56, 9.57, 9.56, 9.63, 9.65], 0.01
    ),  # 时间(s)
    "d_ballDiameter": CalculationUnit(
        "d", [1.985e-3, 1.983e-3, 1.988e-3], 0.004e-3
    ),  # 小球的直径(m)
    "m_ballMass": CalculationUnit("m", 1.625e-3 / 50, 0.001e-3 / 50),  # 1个小球质量(kg)
    "D_pipeDiameter": CalculationUnit(
        "D", [60.60e-3, 60.52e-3, 60.52e-3], 0.02e-3
    ),  # 油桶直径3次(m)
    "s_fallDistance": CalculationUnit(
        "s", [181.8e-3, 180.0e-3, 180.0e-3], 0.5e-3
    ),  # 距离(m)
    "rho1_densityOfOil": CalculationUnit(
        "rho1", 0.9565e3, 0.0010e3
    ),  # 油的密度(kg/m^3)
}

for val in data_dict.values():
    print(val.trueValue())

t = data_dict["t_fallTime"].mean
d = data_dict["d_ballDiameter"].mean
m = data_dict["m_ballMass"].mean
D = data_dict["D_pipeDiameter"].mean
s = data_dict["s_fallDistance"].mean
rho1 = data_dict["rho1_densityOfOil"].mean

u_t = data_dict["t_fallTime"].uncertainty()
u_d = data_dict["d_ballDiameter"].uncertainty()
u_m = data_dict["m_ballMass"].uncertainty()
u_D = data_dict["D_pipeDiameter"].uncertainty()
u_s = data_dict["s_fallDistance"].uncertainty()
u_rho1 = data_dict["rho1_densityOfOil"].uncertainty()

# eta
eta = g / 18 * (6 * m / np.pi / d - rho1 * d**2) * t / s * D / (D + 2.4 * d)

print(eta)

# 偏微分计算
d_part = (-g * D / 18* (6 * m / np.pi * (D + 4.8 * d) / (d**2 * (D + 2.4 * d) ** 2)+ rho1 * (2 * D * d + 2.4 * d**2) / (D + 2.4 * d) ** 2)* t/ s)
D_part = (g / 18 * (6 * m / np.pi / d - rho1 * d**2) * t / s * 2.4 * d / (D + 2.4 * d) ** 2)
s_part = -g / 18 * (6 * m / np.pi / d - rho1 * d**2) * t / s**2 * D / (D + 2.4 * d)
t_part = g / 18 * (6 * m / np.pi / d - rho1 * d**2) / s * D / (D + 2.4 * d)
rho1_part = -g / 18 * t * d**2 / s * D / (D + 2.4 * d)
m_part = g / 18 * (6 / np.pi / d) * t / s * D / (D + 2.4 * d)

# 计算u_eta
u_eta = np.linalg.norm(
    [
        d_part * u_d,
        D_part * u_D,
        s_part * u_s,
        t_part * u_t,
        rho1_part * u_rho1,
        m_part * u_m,
    ]
)
print(u_eta)

# 计算u_r_eta (%)
u_r_eta = u_eta / eta * 100
print(u_r_eta)

parts = {
    "d": d_part,
    "D": D_part,
    "s": s_part,
    "t": t_part,
    "rho1": rho1_part,
    "m": m_part,
}

# 表格

from tabulate import tabulate

# 创建一个字典列表
data = [val.evaluate(parts[val.name]) for val in data_dict.values()]

data.append(
    {
        "name": "eta",
        "mean": eta,
        "sigma": None,
        "delta_A": None,
        "delta_B": None,
        "u": u_eta,
        "u_r": u_r_eta,
        "part_abs": None,
        "part_times_u_abs": None,
    }
)

# 使用tabulate打印表格
print(tabulate(data, headers="keys" ))
print(f"T={T_temperature}")
