from . import angle, mijia


import numpy as np
import cv2
import subprocess


def add_xmp_data(fisheye_path, equir_path):
    rot_mat = np.array(mijia.get_mijia360_gyro(fisheye_path)).reshape(3, 3)
    # because seam is in the middle
    rot_mat = rot_mat.dot(angle.rotateX(-np.pi / 2).T)
    # pitch, roll, yaw
    rot_angles = angle.rotationMatToEuler(rot_mat)
    rot_angles *= 180 / np.pi
    rot_angles *= -1

    rot_angles = np.round(rot_angles, 2)

    img = cv2.imread(equir_path)
    height, width = img.shape[:2]

    # initial view directions - positive rotation
    # pitch : look down
    # roll : lean on right
    # heading : rotate to right
    # note, pose directions require opposite

    # https://developers.google.com/streetview/spherical-metadata

    data = {
        "CaptureSoftware": "MiSphere",
        "StitchingSoftware": "FishPyPano",
        "ProjectionType": "equirectangular",
        "InitialViewHeadingDegrees": 0,
        "PoseHeadingDegrees": rot_angles[2],
        "PosePitchDegrees": rot_angles[0],
        "PoseRollDegrees": rot_angles[1],
        "CroppedAreaImageWidthPixels": width,
        "CroppedAreaImageHeightPixels": height,
        "FullPanoWidthPixels": width,
        "FullPanoHeightPixels": height,
        "CroppedAreaLeftPixels": 0,
        "CroppedAreaTopPixels": 0,
        "SourcePhotosCount": 2
    }

    xmp_tags = ['-{}={}'.format(k, v) for k, v in data.items()]

    xmp_tags = ["exiftool"] + xmp_tags + [equir_path]
    copy_tags = ["exiftool", "-TagsFromFile",
                 fisheye_path, "-all:all", equir_path]
    delete_original = ["exiftool", "-delete_original!", equir_path]

    print(">>", " ".join(copy_tags))
    subprocess.run(copy_tags, check=True)
    print(">>", " ".join(xmp_tags))
    subprocess.run(xmp_tags, check=True)
    print(">>", " ".join(delete_original))
    subprocess.run(delete_original, check=True)
