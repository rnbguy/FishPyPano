import json
import argparse
import util
import tqdm

import numpy as np

from scipy.optimize import basinhopping


def dual_norm_fisheye_to_3d_umeyama(norm_2d_points):
    norm_2d_points = norm_2d_points.reshape((-1, 2))
    all_3d_points = util.normalized_fisheye_to_3d(
        norm_2d_points, np.pi)
    _3d_points = all_3d_points.reshape((-1, 2, 3))
    return util.umeyama(
        _3d_points[:, 1, :],
        _3d_points[:, 0, :]
    )


def get_umeyama(params):
    c = params[:4].reshape((2, 2))
    norm_points = (points - c) / params[4]
    norm_points = util.undistort(norm_points)
    return dual_norm_fisheye_to_3d_umeyama(norm_points)


def loss_func(params):
    _, e = get_umeyama(params)
    return e * 1000000


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--points", help="path to matched points", required=True)
args = ap.parse_args()

with open(args.points, 'r') as jh:
    points = np.array(json.load(jh))


mijia_fisheye_size = 3456

points[:, 1, 0] -= mijia_fisheye_size

bounds = np.array([
    [1, mijia_fisheye_size],
    [1, mijia_fisheye_size],
    [1, mijia_fisheye_size],
    [1, mijia_fisheye_size],
    [1, mijia_fisheye_size]
])


minimizer_kwargs = {
    "method": "L-BFGS-B",
    "bounds": bounds
}

n_iter = 300

init_params = bounds.mean(axis=1)

with tqdm.tqdm(total=300) as pbar:
    def pbar_update(*X):
        global pbar
        pbar.update(1)

    ret = basinhopping(loss_func,
                       init_params,
                       minimizer_kwargs=minimizer_kwargs,
                       callback=pbar_update,
                       disp=False,
                       niter=n_iter)

opt_params = ret.x

T, e = get_umeyama(opt_params)

T = np.linalg.inv(T.T)[:, :3]

data = {
    'fisheye': opt_params.tolist(),
    'transform_3d': T.tolist()
}

with open('calib_data.json', 'w') as jh:
    json.dump(data, jh, indent=4)

print("Saved calibration data at -- {}".format('calib_data.json'))
