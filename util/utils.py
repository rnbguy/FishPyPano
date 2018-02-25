import numpy as np
import cv2


def crop_fisheye_img(fimg, circle, radius):
    circle = np.round(circle).astype(int)
    radius = np.round(radius).astype(int)

    size = fimg.shape[0]

    circle_x, circle_y = circle
    crop_l = int(np.ceil(circle_x - radius))
    crop_t = int(np.ceil(circle_y - radius))
    crop_r = int(np.floor(circle_x + radius))
    crop_b = int(np.floor(circle_y + radius))
    return cv2.resize(fimg[crop_t:crop_b, crop_l:crop_r], (size, size))


def add_alpha_channel(img):
    height, width, channels = img.shape
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    circle_y = height >> 1
    circle_x = width >> 1
    circle_r = min(circle_x, circle_y)
    mask = np.zeros((height, width, 4)).astype(img.dtype)
    color = (255, 255, 255, 255)
    cv2.circle(mask, (circle_x, circle_y), circle_r, color, -1)
    img_ = cv2.bitwise_and(img, mask)
    return img_


def get_seam(energy_map):
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
