import numpy as np


def normalized_fisheye_to_3d(xy_f, hfov):
    r = np.linalg.norm(xy_f, axis=-1)
    theta = np.arctan2(xy_f[..., 1], xy_f[..., 0])
    phi = r * hfov / 2

    x_3d = np.sin(phi) * np.cos(theta)
    y_3d = np.cos(phi)
    z_3d = np.sin(phi) * np.sin(theta)

    vec = np.dstack([x_3d, y_3d, z_3d]).reshape(-1, 3)

    return vec


def normalized_3d_to_fisheye(xy_3d, hfov):
    x_3d, y_3d, z_3d = np.hsplit(xy_3d, 3)

    phi = np.arccos(y_3d)
    theta = np.arctan2(z_3d, x_3d)

    r = 2 * phi / hfov
    xy_f = np.dstack([np.cos(theta), np.sin(theta)]).reshape(-1, 2)
    xy_f = xy_f * r

    return xy_f


def normalized_3d_to_equirectangle(xyz_3d):
    x_3d, y_3d, z_3d = np.hsplit(xyz_3d, 3)

    longitude = np.arctan2(x_3d, y_3d)
    latitude = np.arcsin(z_3d)

    x_e = longitude / np.pi
    y_e = 2 * latitude / np.pi

    return np.dstack([x_e, y_e]).reshape(-1, 2)


def normalized_equirectangle_to_3d(xy_e):
    x_e, y_e = np.hsplit(xy_e, 2)

    longitude = x_e * np.pi
    latitude = y_e * np.pi / 2

    sin_lati = np.sin(latitude)
    cos_lati = np.cos(latitude)
    sin_long = np.sin(longitude)
    cos_long = np.cos(longitude)

    vec = np.dstack([cos_lati * sin_long, cos_lati * cos_long, sin_lati])
    vec = vec.reshape(-1, 3)

    return vec
