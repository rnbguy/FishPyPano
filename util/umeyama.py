import numpy as np


def umeyama(src, dst):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    e : Squared error sum
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num, dim = src.shape

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones(dim, dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    src_varsum = src_demean.var(axis=0).sum()
    dst_varsum = dst_demean.var(axis=0).sum()

    Sd = np.dot(S, d)

    # Eq. (41) and (42).
    scale = Sd / src_varsum

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    e = dst_varsum - (Sd ** 2) / src_varsum

    return T, e


if __name__ == "__main__":
    # Run an example test
    # We have 3 points in 3D. Every point is a column vector of this matrix A
    A = np.random.rand(5, 3)
    # Deep copy A to get B
    B = A.copy()
    # and sum a translation on z axis (3rd row) of 10 units
    B[:, 2] = B[:, 2] + 10

    # Reconstruct the transformation with ralign.ralign
    T, e = umeyama(A, B)
    assert(np.allclose(
        np.pad(B, ((0, 0), (0, 1)), 'constant', constant_values=1),
        np.pad(A, ((0, 0), (0, 1)), 'constant', constant_values=1).dot(T.T)))
    print("Similarity matrix with error", e)
    print(T)
