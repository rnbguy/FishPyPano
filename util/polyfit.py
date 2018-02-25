import numpy as np


def bezier(p):
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    return 3 * (1 - t) * t**2 * p + t**3 * np.ones(2)


def bezier_fit(p):
    t = np.linspace(1e-32, 2, 1e5).reshape(-1, 1)
    bez = ((3 * (1 - t) * t**2) * p) + ((t**3) * np.ones(2))
    x = bez[:, 0]
    y = bez[:, 1]
    b = np.polyfit(x, y / x, 3)
    b /= b.sum()
    b = np.r_[b, [0]]
    return b


def inv_polyfit(b):
    x = np.linspace(1e-32, 2, 1e5)
    y = np.polyval(b, x)
    a = np.polyfit(y, x / y, len(b) - 2)
    assert(b.size == a.size + 1)
    return np.r_[a, [0]]
