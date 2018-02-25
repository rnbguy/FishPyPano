import numpy as np


def undistort_tan(r, k1, k2):
    return np.tan(k1 * r) / k2


def distort_tan(r, k1, k2):
    return np.arctan(k2 * r) / k1


def _distort(dist_points, func):
    old_shape = dist_points.shape
    dist_points = dist_points.reshape((-1, 2))
    r = np.linalg.norm(dist_points, axis=1)
    t = np.arctan2(dist_points[:, 1], dist_points[:, 0])
    r_ = func(r)
    x_ = r_ * np.cos(t)
    y_ = r_ * np.sin(t)
    return np.c_[x_, y_].reshape(old_shape)


k = .76


def undistort(dist_points):
    return _distort(
        dist_points,
        lambda r: undistort_tan(r, k, k)
    )


def distort(dist_points):
    return _distort(
        dist_points,
        lambda r: distort_tan(r, k, k)
    )
