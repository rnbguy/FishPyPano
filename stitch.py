import cv2
import numpy as np
import argparse
import util
import json
import os


def fisheye_img_to_equirect(img, hfov, center, trans_mat=None):
    height, width = img.shape[:2]
    x_lim = height << 1
    y_lim = height

    xy_e = np.stack(np.indices((x_lim, y_lim)), -1).astype(np.double)
    xy_e = xy_e.reshape(-1, xy_e.shape[-1])

    xy_e = (xy_e / [x_lim / 2, y_lim / 2]) - 1
    xy_3d = util.normalized_equirectangle_to_3d(xy_e)
    xy_3d = xy_3d.dot(util.rotateX(-np.pi / 2))
    if trans_mat is not None:
        pad_xy = np.pad(xy_3d,
                        ((0, 0), (0, 1)),
                        'constant',
                        constant_values=1)
        xy_3d = pad_xy.dot(trans_mat)
        xy_3d /= np.linalg.norm(xy_3d, axis=1).reshape((-1, 1))
    xy_f = util.normalized_3d_to_fisheye(xy_3d, hfov)
    # do distortion here
    xy_f = util.distort(xy_f)
    # xy_f is ndarry of (x, y) points in [-1, 1]x[-1,1]
    xy_f *= radius
    xy_f += center

    xy_f = xy_f.reshape((x_lim, y_lim, 2))

    # equirectangular projection
    equi = cv2.remap(img,
                     xy_f[..., 0].T.astype(np.float32),
                     xy_f[..., 1].T.astype(np.float32),
                     cv2.INTER_CUBIC,
                     cv2.BORDER_REFLECT)

    return equi


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="path to the dual fisheye image", required=True)
ap.add_argument("-c", "--calib",
                help="path to calib data", required=True)
ap.add_argument("-d", "--debug",
                help="save all debug images",
                action='store_true')
args = ap.parse_args()


orig_img = cv2.imread(args.image)

img_size = orig_img.shape[0]

with open(args.calib) as fh:
    data = json.load(fh)

circles = np.array(data["fisheye"][:4]).reshape((2, 2))
radius = data["fisheye"][4]
trans_mat = np.array(data["transform_3d"])

print("Loaded calibration data..")

gyro_rot = np.array(util.get_mijia360_gyro(args.image)).reshape(3, 3)
gyro_rot[:, [1, 2]] = gyro_rot[:, [2, 1]]
gyro_rot[[1, 2], :] = gyro_rot[[2, 1], :]

img1, img2 = np.hsplit(cv2.imread(args.image), 2)

img1 = util.add_alpha_channel(img1)
img2 = util.add_alpha_channel(img2)

if args.debug:
    cv2.imwrite("fisheye_alpha.png", np.hstack([img1, img2]))

print("Projecting fisheye images to equirectangular projection..")
print("Projecting left fisheye image..")
equi1 = fisheye_img_to_equirect(img1, np.pi, circles[0])
print("Projecting right fisheye image..")
equi2 = fisheye_img_to_equirect(img2, np.pi, circles[1], trans_mat)

if args.debug:
    cv2.imwrite('equi1.png', equi1)
    cv2.imwrite('equi2.png', equi2)

print("Done projecting, starting stitching..")

zero_channel1 = np.isclose(equi1[..., 3], 0)  # only channel2 active
zero_channel2 = np.isclose(equi2[..., 3], 0)  # only channel1 active

equi1[zero_channel1] = [0, 0, 0, 0]
equi2[zero_channel2] = [0, 0, 0, 0]

zero_channel = np.logical_or(zero_channel1, zero_channel2)

overlap1 = equi1.copy()
overlap2 = equi2.copy()

overlap1[zero_channel] = [0, 0, 0, 0]
overlap2[zero_channel] = [0, 0, 0, 0]

overlap_y_min = np.where(zero_channel == 0)[0].min()
overlap_y_max = np.where(zero_channel == 0)[0].max()

src = equi1.copy()
dst = equi2.copy()
dst[zero_channel2] = src[zero_channel2]

overlayed = dst.copy()

if args.debug:
    cv2.imwrite("equi_overlayed.png", dst)

overlap_y_min = np.where(zero_channel2)[0].max()
overlap_y_max = np.where(zero_channel1)[0].min()

overlap_y_mid = (overlap_y_max + overlap_y_min) / 2
overlap_y_gap = (overlap_y_max - overlap_y_min) / 6

overlap_y_min = int(round(overlap_y_mid - overlap_y_gap))
overlap_y_max = int(round(overlap_y_mid + overlap_y_gap))

equi1[overlap_y_max:, ...] = 0
equi2[:overlap_y_min, ...] = 0

src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2BGR)

equi1_overlap = equi1[overlap_y_min:overlap_y_max, ..., :3]
equi2_overlap = equi2[overlap_y_min:overlap_y_max, ..., :3]

if args.debug:
    cv2.imwrite("overlap1.jpg", equi1_overlap)
    cv2.imwrite("overlap2.jpg", equi2_overlap)
    cv2.imwrite("diff_orig_equi.jpg", cv2.absdiff(
        equi1_overlap, equi2_overlap))

print("Calculating optical flow and Warping..")
warped_overlap1, warped_overlap2 = util.flow_warp(equi1_overlap, equi2_overlap)

equi1[overlap_y_min:overlap_y_max, ..., :3] = warped_overlap1
equi2[overlap_y_min:overlap_y_max, ..., :3] = warped_overlap2

if args.debug:
    cv2.imwrite("warped_equi1.png", equi1)
    cv2.imwrite("warped_equi2.png", equi2)

print("Finding best seam..")
diff = cv2.absdiff(warped_overlap1, warped_overlap2)
diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

if args.debug:
    cv2.imwrite("diff_warped_equi.jpg", diff)

seam = np.array(util.get_seam(diff))
seam = np.fliplr(seam)

diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

cv2.polylines(diff, [seam], False, (0, 0, 255), 2)

if args.debug:
    cv2.imwrite("equi_diff_seam.jpg", diff)

seam[:, 1] += overlap_y_min

max_y = np.amax(seam[:, 1])
min_y = np.amin(seam[:, 1])

equi1 = cv2.cvtColor(equi1, cv2.COLOR_BGRA2BGR)
equi2 = cv2.cvtColor(equi2, cv2.COLOR_BGRA2BGR)

equi2[:overlap_y_min] = equi1[:overlap_y_min]

src_mask = np.zeros_like(equi1)
poly = np.array([
    [0, 0],
    [equi1.shape[0] * 2 - 1, 0],
],
)
poly = np.vstack([poly, seam])
cv2.fillPoly(src_mask, [poly], (255, 255, 255))
if args.debug:
    cv2.imwrite('src_mask.jpg', src_mask)
center = (equi1_overlap.shape[1] // 2, max_y // 2 + 1)

print("Seamless stitching..")
output = cv2.seamlessClone(
    equi1, equi2, src_mask, center, cv2.NORMAL_CLONE)

stitched_path = "360_" + os.path.basename(args.image)

cv2.imwrite(stitched_path, output)

print("Adding Google Photo Sphere XMP Metadata..")

util.add_xmp_data(args.image, stitched_path)

print("Stitchied image saved at -- {}".format(stitched_path))
