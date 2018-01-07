import argparse
import cv2
import numpy as np
from scipy.optimize import basinhopping
import json
from PIL import Image
import struct


class DualSphereImg:
    calib_data_json_path = "calibrated_data.json"
    radius = 3356 >> 1

    def __init__(self, img_path):
        self.img_path = img_path
        self.dual_img = cv2.imread(img_path)
        img_shape = self.dual_img.shape
        assert(img_shape[0] << 1 == img_shape[1])
        self.fisheye_size = img_shape[0]
        self.left_frame, self.right_frame = np.hsplit(self.dual_img, 2)
        self.mathed_points = np.array(
            [
                [[521, 2684], [6437, 2725]],
                [[2765, 2924], [4131, 2903]],
                [[3216, 1509], [3551, 1459]],
                [[3264, 1219], [3719, 1216]],
                [[2447, 276], [4507, 338]],
                [[1890, 112], [5040, 202]],
                [[192, 1986], [6747, 1985]],
                [[324, 1202], [6715, 1146]],
                [[1026, 3133], [5883, 3142]],
                [[2282, 216], [4658, 262]]
            ]
        )

    def rotateX(self, a):
        return np.array([1, 0, 0,
                         0, np.cos(a), -np.sin(a),
                         0, np.sin(a), np.cos(a)]).reshape(3,  3)

    def rotateY(self, a):
        return np.roll(np.roll(self.rotateX(a), 1, -2), 1, -1)

    def rotateZ(self, a):
        return np.roll(np.roll(self.rotateX(a), 2, -2), 2, -1)

    def get_rot_basis(self, angles):
        x, y, z = angles
        mat = self.rotateX(x)
        mat = self.rotateY(y).dot(mat)
        mat = self.rotateZ(z).dot(mat)
        return mat

    def load_calibration(self):
        with open(self.calib_data_json_path) as f:
            json_data = json.load(f)

        self.cir0_c = np.array(json_data['left_circle']['center']).astype(int)
        self.cir1_c = np.array(json_data['right_circle']['center']).astype(int)
        self.hfov0 = json_data['left_circle']['hfov'] * np.pi / 180
        self.hfov1 = json_data['right_circle']['hfov'] * np.pi / 180
        self.err_rot = self.get_rot_basis(
            np.array(json_data['right_circle']['rotate']) * np.pi / 180)
        self.err_trans = np.array(json_data['right_circle']['translate'])
        self.param = np.append(json_data['left_circle']['center'],
                               json_data['right_circle']['center'])
        self.param = np.append(
            self.param, [json_data['left_circle']['hfov'], json_data['right_circle']['hfov']])
        self.param = np.append(self.param, np.array(json_data['right_circle']['rotate']))
        self.param = np.append(self.param, np.array(json_data['right_circle']['translate']))

    def get_rotatation_basis(self):
        im = Image.open(self.img_path)
        exif_data = im._getexif()
        USERCOMMENT = 37510
        raw_data = exif_data[USERCOMMENT]
        self.gyroBasis = np.array([e[0] for e in struct.iter_unpack('f', raw_data)]).reshape(-1,  3)

    def parse_rotation_basis(self, data_string):
        raw_data = bytearray.fromhex(data_string)
        self.gyroBasis = np.array([e[0] for e in struct.iter_unpack('f', raw_data)]).reshape(-1, 3)

    def rotation_basis_to_angle(self):
        yaw = np.arctan2(self.gyroBasis[1, 0],
                         self.gyroBasis[0, 0])
        roll = np.arctan2(-self.gyroBasis[2, 0],
                          np.linalg.norm([
                              self.gyroBasis[2, 1],
                              self.gyroBasis[2, 2]]))
        pitch = np.arctan2(self.gyroBasis[2, 1],
                           self.gyroBasis[2, 2])
        return np.array([pitch, roll, yaw])

    def normalized_fisheye_to_3d(self, xy_f, hfov):
        r = np.linalg.norm(xy_f, axis=-1)
        theta = np.arctan2(xy_f[..., 1], xy_f[..., 0])
        phi = r * hfov / 2

        x_3d = np.sin(phi) * np.cos(theta)
        y_3d = np.sin(phi) * np.sin(theta)
        z_3d = np.cos(phi)

        vec = np.dstack([x_3d, y_3d, z_3d]).reshape(-1, 3)
        return vec

    def normalized_3d_to_fisheye(self, xy_3d, hfov):
        x_3d, y_3d, z_3d = np.hsplit(xy_3d, 3)

        phi = np.arccos(z_3d)
        theta = np.arctan2(y_3d, x_3d)

        r = 2 * phi / hfov
        xy_f = np.dstack([np.cos(theta), np.sin(theta)]).reshape(-1, 2)
        xy_f = xy_f * r

        return xy_f

    def normalized_equirectangle_to_3d(self, xy_e):
        x_e, y_e = np.hsplit(xy_e, 2)

        longitude = x_e * np.pi
        latitude = y_e * np.pi / 2

        sin_lati = np.sin(latitude)
        cos_lati = np.cos(latitude)
        sin_long = np.sin(longitude)
        cos_long = np.cos(longitude)

        vec = np.dstack([cos_lati * sin_long, sin_lati, cos_lati * cos_long])
        vec = vec.reshape(-1, 3)

        return vec

    def normalized_3d_to_equirectangle(self, xyz_3d):
        x_3d, y_3d, z_3d = np.hsplit(xyz_3d, 3)

        longitude = np.arctan2(x_3d, z_3d)
        latitude = np.arcsin(y_3d)

        x_e = longitude / np.pi
        y_e = 2 * latitude / np.pi

        return np.dstack([x_e, y_e]).reshape(-1, 2)

    def undo_movements(self, vec, is_back):
        # vec = vec.dot(self.gyroBasis) # disables for stitiching

        vec = vec.dot(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

        if is_back:
            # undo fix rotate
            # since err_rot is orthogonal matrix, transpose == inv
            vec = vec.dot(self.err_rot)
            # undo fix translate
            vec = vec - self.err_trans
            vec = vec / np.linalg.norm(vec, axis=1).reshape(-1, 1)
            # rotate 180 deg around y axis
            vec *= [-1, 1, -1]

        return vec

    def map_norm_fisheye_to_3d(self, xy_f, hfov, is_back, trans, rot):
        vec = self.normalized_fisheye_to_3d(xy_f, hfov)

        if is_back:
            vec *= [-1, 1, -1]
        if trans is not None:
            vec = vec + trans
            vec = vec / np.linalg.norm(vec, axis=-1)[..., None]
        if rot is not None:
            rotMat = self.get_rot_basis(rot)
            vec = vec.dot(rotMat.T)

        return vec

    def calibration_func(self, params, display=False):
        circ_c0 = params[0:2]
        circ_c1 = params[2:4]
        hfov0 = params[4] * np.pi / 180
        hfov1 = params[5] * np.pi / 180
        rot = params[6:9] * np.pi / 180
        trans = params[9:12]

        points0 = (self.mathed_points[:, 0, :] - circ_c0) / self.radius
        points1 = (self.mathed_points[:, 1, :] - circ_c1) / self.radius

        _3d0 = self.map_norm_fisheye_to_3d(points0, hfov0, False, None, None)
        _3d1 = self.map_norm_fisheye_to_3d(points1, hfov1, True, trans, rot)

        if display:
            norm_3d0 = self.normalized_3d_to_equirectangle(_3d0)
            norm_3d1 = self.normalized_3d_to_equirectangle(_3d1)
            equi0 = (norm_3d0 + 1) * [self.fisheye_size, self.fisheye_size // 2]
            equi1 = (norm_3d1 + 1) * [self.fisheye_size, self.fisheye_size // 2]
            equi1[:, 0] += self.fisheye_size
            x_lim = self.fisheye_size << 1
            y_lim = self.fisheye_size

            ###
            xy_e = (equi1 / [self.fisheye_size, self.fisheye_size // 2]) - 1
            xy_3d = self.normalized_equirectangle_to_3d(xy_e)
            xy_3d = self.undo_movements(xy_3d, True)
            xy_f = self.normalized_3d_to_fisheye(xy_3d, hfov1)

        return np.linalg.norm(_3d0 - _3d1, axis=-1).sum(axis=-1) * self.fisheye_size / 2

    def get_matched_points(self, point_json_path):
        if point_json_path is None:
            point_json_path = "dual_points.json"
        with open(point_json_path) as f:
            self.mathed_points = np.array(json.load(f))

    def do_calibration(self, n_iter=150, json_path=None):
        self.get_matched_points(json_path)
        bounds = [
            [self.fisheye_size * .2, self.fisheye_size * .8],
            [self.fisheye_size * .2, self.fisheye_size * .8],
            [self.fisheye_size * 1.2, self.fisheye_size * 1.8],
            [self.fisheye_size * .2, self.fisheye_size * .8],
            [180, 250],
            [180, 250],
            [-90, 90],
            [-90, 90],
            [-90, 90],
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ]
        init_param = np.array([
            self.fisheye_size * .5,
            self.fisheye_size * .5,
            self.fisheye_size * 1.5,
            self.fisheye_size * .5,
            190,
            190,
            0,
            0,
            0,
            0,
            0,
            0,
        ])

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        ret = basinhopping(self.calibration_func,
                           init_param,
                           minimizer_kwargs=minimizer_kwargs,
                           disp=True,
                           niter=n_iter)
        param = list(ret.x)
        left_circle = dict()
        right_circle = dict()
        left_circle['center'] = param[0:2]
        right_circle['center'] = param[2:4]
        left_circle['hfov'] = param[4]
        right_circle['hfov'] = param[5]
        right_circle['rotate'] = param[6:9]
        right_circle['translate'] = param[9:12]
        self.param = param

        j_data = dict()
        j_data['left_circle'] = left_circle
        j_data['right_circle'] = right_circle
        with open(self.calib_data_json_path, 'w') as f:
            json.dump(j_data, f, indent=4)

    def crop_fisheye_img(self, fimg, circle):
        circle_x, circle_y = circle
        crop_l = int(np.ceil(circle_x - self.radius))
        crop_t = int(np.ceil(circle_y - self.radius))
        crop_r = int(np.floor(circle_x + self.radius))
        crop_b = int(np.floor(circle_y + self.radius))
        return cv2.resize(fimg[crop_t:crop_b, crop_l:crop_r],
                          (self.fisheye_size, self.fisheye_size))

    def add_alpha_channel(self, img):
        height, width, channels = img.shape
        if channels == 3:
            b, g, r = cv2.split(img)
            a = np.full(r.shape,  255).astype(r.dtype)
            img = cv2.merge((b, g, r, a))
        circle_y = height >> 1
        circle_x = width >> 1
        circle_r = min(circle_x, circle_y)
        mask = np.zeros((height, width, 4)).astype(img.dtype)
        color = (255, 255, 255, 255)
        cv2.circle(mask, (circle_x, circle_y), circle_r, color, -1)
        img_ = cv2.bitwise_and(img, mask)
        return img_

    def do_movements(vec, is_back):
        if is_back:
            # rotate 180 deg around y axis
            vec *= [-1, 1, -1]
            # fix translate
            vec = vec + self.err_trans
            vec = vec / np.linalg.norm(vec, axis=1).reshape(-1, 1)
            # fix rotate
            vec = vec.dot(self.err_rot.T)

        vec = vec.dot(self.gyroBasis.T)

        return vec

    def fisheye_img_to_equirect(self, img, hfov, is_back=False):
        x_lim = self.fisheye_size << 1
        y_lim = self.fisheye_size

        ###
        xy_e = np.stack(np.indices((x_lim, y_lim)), -1)
        xy_e = xy_e.reshape(-1, xy_e.shape[-1])

        xy_e = (xy_e / [x_lim / 2, y_lim / 2]) - 1
        xy_3d = self.normalized_equirectangle_to_3d(xy_e)
        xy_3d = self.undo_movements(xy_3d, is_back)
        xy_f = self.normalized_3d_to_fisheye(xy_3d, hfov)
        # xy_f is ndarry of (x, y) points in [-1, 1]x[-1,1]
        xy_f = (xy_f + 1) * (self.fisheye_size / 2)

        xy_f = xy_f.reshape((x_lim, y_lim, 2))

        dst = cv2.remap(img,
                        xy_f[..., 0].T.astype(np.float32),
                        xy_f[..., 1].T.astype(np.float32),
                        cv2.INTER_CUBIC,
                        cv2.BORDER_REFLECT)
        ###

        return dst

    def get_seam(self, energy_map):
        t_shape = energy_map.shape[:2]
        dp_t = np.empty(t_shape + (2,), int)
        for x in range(t_shape[1]):
            for y in range(t_shape[0]):
                dp_t[y][x][0] = np.linalg.norm(energy_map[y][x])
                if x > 0:
                    prev_energy = dp_t[y, x - 1, 0]
                    prev_y = y
                    if y > 0:
                        curr_energy = dp_t[y - 1, x - 1, 0]
                        if curr_energy < prev_energy:
                            prev_energy = curr_energy
                            prev_y = y - 1
                    if y + 1 < t_shape[0]:
                        curr_energy = dp_t[y + 1, x - 1, 0]
                        if curr_energy < prev_energy:
                            prev_energy = curr_energy
                            prev_y = y + 1
                    dp_t[y, x, 0] += prev_energy
                    dp_t[y, x, 1] = prev_y
        min_seam = np.argmin(dp_t[:, -1, 0])
        pts = [[min_seam, t_shape[1] - 1]]
        for i in range(t_shape[1] - 1):
            prev_y, prev_x = pts[-1]
            x = prev_x - 1
            y = dp_t[prev_y, prev_x, 1]
            pts.append([y, x])
        return pts

    def warp_img(self, do_calibrate=False):
        self.load_calibration()

        self.get_rotatation_basis()

        img0 = self.crop_fisheye_img(self.dual_img, self.cir0_c)
        img1 = self.crop_fisheye_img(self.dual_img, self.cir1_c)

        img0 = self.add_alpha_channel(img0)
        img1 = self.add_alpha_channel(img1)

        equi0 = self.fisheye_img_to_equirect(img0, self.hfov0, False)
        equi1 = self.fisheye_img_to_equirect(img1, self.hfov1, True)

        return (equi0, equi1)

    def seamlessStitch(self, equi0, equi1):
        zero_channel0 = np.isclose(equi0[..., 3], 0)  # only channel2 active
        zero_channel1 = np.isclose(equi1[..., 3], 0)  # only channel1 active

        equi0[zero_channel0] = [0, 0, 0, 0]
        equi1[zero_channel1] = [0, 0, 0, 0]

        src = equi0.copy()
        dst = equi1.copy()
        dst[zero_channel1] = src[zero_channel1]

        cv2.imwrite("equi_overlayed.png", dst)

        overlap_y_min = np.where(zero_channel1)[0].max()
        overlap_y_max = np.where(zero_channel0)[0].min()

        src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2BGR)

        equi0_overlap = equi0[overlap_y_min:overlap_y_max, ..., :3]
        equi1_overlap = equi1[overlap_y_min:overlap_y_max, ..., :3]

        diff = cv2.absdiff(equi0_overlap, equi1_overlap)

        seam = np.array(self.get_seam(diff))
        seam = np.fliplr(seam)

        cv2.polylines(diff, [seam], False, (0, 0, 255), 2)
        cv2.imwrite("equi_diff_seam.jpg", diff)

        seam[:, 1] += overlap_y_min

        max_y = np.amax(seam[:, 1])
        min_y = np.amin(seam[:, 1])

        poly = np.array([
            [0, 0],
            [self.fisheye_size * 2 - 1, 0],
        ],
        )
        src_mask = np.zeros(src.shape, src.dtype)
        poly = np.vstack([poly, seam])
        cv2.fillPoly(src_mask, [poly], (255, 255, 255))
        center = (self.fisheye_size, max_y // 2 + 1)
        output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite("equi_stitched.jpg", output)
        return output

    def rotate_imge(self, img, rotMat):
        x_lim = self.fisheye_size << 1
        y_lim = self.fisheye_size

        ###
        xy_s = np.stack(np.indices((x_lim, y_lim)), -1)
        xy_s = xy_s.reshape(-1, xy_s.shape[-1])

        xy_s = (xy_s / [x_lim / 2, y_lim / 2]) - 1
        xy_3d = self.normalized_equirectangle_to_3d(xy_s)
        xy_3d = xy_3d.dot(rotMat)
        xy_s = self.normalized_3d_to_equirectangle(xy_3d)
        xy_s = (xy_s + 1) * [(x_lim - 1) / 2, (y_lim - 1) / 2]

        print(xy_s[:, 0].min())
        print(xy_s[:, 0].max())
        print(xy_s[:, 1].min())
        print(xy_s[:, 1].max())

        xy_s = xy_s.reshape((x_lim, y_lim, 2))

        dst = cv2.remap(img,
                        xy_s[..., 0].T.astype(np.float32),
                        xy_s[..., 1].T.astype(np.float32),
                        cv2.INTER_NEAREST,
                        )
        ###

        return dst

    def rotate_correctly(self, img):

        gyroRot = self.gyroBasis

        gyroRot[:, [1, 2]] = gyroRot[:, [2, 1]]
        gyroRot[[1, 2], :] = gyroRot[[2, 1], :]

        rotMat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        rotMat = np.dot(gyroRot, rotMat)

        # we want to move our view direction from 3d vector V to rotMat.dot(V)

        correct_gyro_img = self.rotate_imge(img, rotMat)

        return correct_gyro_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image", required=True)
    ap.add_argument("-c", "--calibrate", help="calibrate with points", action='store_true')
    ap.add_argument("-p", "--points", help="json list of matched points")
    args = ap.parse_args()
    if args.calibrate:
        dual = DualSphereImg(args.image)
        dual.do_calibration(350, args.points)
    else:
        dual = DualSphereImg(args.image)
        img0, img1 = dual.warp_img()
        cv2.imwrite("equi0.png", img0)
        cv2.imwrite("equi1.png", img1)
        equirect_img = dual.seamlessStitch(img0, img1)
        correct_gyro_img = dual.rotate_correctly(equirect_img)
        cv2.imwrite("equi_rotated.jpg", correct_gyro_img)
