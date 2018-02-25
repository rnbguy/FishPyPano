import numpy as np


def rotateX(a):
    return np.array([1, 0, 0,
                     0, np.cos(a), -np.sin(a),
                     0, np.sin(a), np.cos(a)]).reshape(3,  3)


def rotateY(a):
    return np.roll(np.roll(rotateX(a), 1, -2), 1, -1)


def rotateZ(a):
    return np.roll(np.roll(rotateX(a), 2, -2), 2, -1)


def get_rot_basis(angles):
    x, y, z = angles
    mat = rotateX(x)
    mat = rotateY(y).dot(mat)
    mat = rotateZ(z).dot(mat)
    return mat


def rotationMatToEuler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
