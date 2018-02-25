import cv2
import numpy as np
import json


def draw_circles(img, circles, radius):
    circles = np.round(circles).astype(int)
    radius = np.round(radius).astype(int)
    c1, c2 = circles
    img1, img2 = np.hsplit(img, 2)
    cv2.circle(img1, tuple(c1), radius, (0, 0, 255), 2)
    cv2.circle(img1, tuple(c1), 2, (0, 0, 255), 20)
    cv2.circle(img2, tuple(c2), radius, (0, 0, 255), 2)
    cv2.circle(img2, tuple(c2), 2, (0, 0, 255), 20)
    return np.hstack([img1, img2])


def draw_points(img, points):
    img_ = img.copy()
    for e in points.astype(int):
        cv2.line(img_, tuple(e[0]), tuple(e[1]), (255, 0, 0), 2)
    return img_


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", help="fisheye image path", required=True)
    ap.add_argument("-p", help="dual points path", required=True)
    ap.add_argument("-c", help="calibration data path", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.i)

    img_size = img.shape[0]

    with open(args.c) as fh:
        data = json.load(fh)

    circles = np.array(data["fisheye"][:4]).reshape((2, 2))
    radius = img_size >> 1

    img = draw_circles(img, circles, radius)

    with open(args.p, 'r') as jh:
        points = np.array(json.load(jh))

    img = draw_points(img, points)

    cv2.imwrite("img_circ_points.jpg", img)
