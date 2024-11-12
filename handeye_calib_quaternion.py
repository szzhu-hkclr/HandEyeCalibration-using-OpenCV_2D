import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd
import math

def find_chessboard_corners(images, pattern_size, ShowCorners=False):
    """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
    chessboard_corners = []
    IndexWithImg = []
    i = 0
    print("Finding corners...")
    for image in images:
        print(">>>>>>>>>>", i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
        if ret:
            chessboard_corners.append(corners)

            cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            if ShowCorners:
                plt.imshow(image)
                plt.title("Detected corner in image: " + str(i))
                plt.show()

            IndexWithImg.append(i)
            i = i + 1
        else:
            print("No chessboard found in image: ", i)
            i = i + 1
    return chessboard_corners, IndexWithImg


def calculate_intrinsics(chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
    """Calculates the intrinsic camera parameters fx, fy, cx, cy from the images"""
    imgpoints = chessboard_corners
    objpoints = []
    for i in range(len(IndexWithImg)):
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)

    print("The projection error from the calibration is: ",
          calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowProjectError))
    print(mtx)
    return mtx


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowPlot=False):
    """Calculates the reprojection error of the camera for each image"""
    total_error = 0
    num_points = 0
    errors = []

    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        errors.append(error)
        total_error += error
        num_points += 1

    mean_error = total_error / num_points
    return mean_error


def compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, Testing=False):
    """Takes the chessboard corners and computes the camera poses"""
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    RTarget2Cam = []
    TTarget2Cam = []
    i = 1
    for corners in chessboard_corners:
        _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)
        if Testing:
            print("Current iteration: ", i, " out of ", len(chessboard_corners[0]), " iterations.")
            print("rvec[0]: ", rvec[0])
            print("rvec[1]: ", rvec[1])
            print("rvec[2]: ", rvec[2])
            print("tvec[0]: ", tvec[0])
            print("tvec[1]: ", tvec[1])
            print("tvec[2]: ", tvec[2])
        i = 1 + i
        R, _ = cv2.Rodrigues(rvec)
        RTarget2Cam.append(R)
        TTarget2Cam.append(tvec)

    return RTarget2Cam, TTarget2Cam


# Example image directory
image_folder = "./2024-11-08"
image_files = sorted(glob.glob(f'{image_folder}/img_*.png'))
images = [cv2.imread(f) for f in image_files]

pattern_size = (11, 8)
square_size = 15 / 1000
ShowProjectError = True
ShowCorners = False

# Camera calibration
chessboard_corners, IndexWithImg = find_chessboard_corners(images, pattern_size, ShowCorners=ShowCorners)
intrinsic_matrix = calculate_intrinsics(chessboard_corners, IndexWithImg,
                                        pattern_size, square_size,
                                        images[0].shape[:2][::-1], ShowProjectError=ShowProjectError)

# Calculate camera extrinsics
RTarget2Cam, TTarget2Cam = compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix)

# Poses with translation and quaternion data
# Each pose is now: [tx, ty, tz, qx, qy, qz, qw]
poses = [
    [0.15051, -0.98167, 0.19784, -0.52527, 0.84827, -0.064261, -0.019863],       #img_16-41-56
    [0.37439, -0.93348, 0.15096, -0.63015, 0.74166, 0.018804, -0.22911],       #img_16-43-58
    [-0.019255, -1.0245, 0.12078, 0.71173, -0.67307, 0.15944, -0.12241],       #img_16-45-42
    [0.1411, -0.84013, 0.10857, 0.97367, 0.047694, -0.042388, 0.21887],       #img_16-47-00
    [0.25611, -1.1328, 0.14794, 0.93581, 0.31917, -0.050942, -0.14071],       #img_16-50-16
    [0.32091, -1.1022, 0.17871, 0.86571, 0.47199, -0.086662, -0.14233],       #img_16-56-18
    [0.26304, -0.88579, 0.18963, 0.98747, 0.038113, -0.11454, 0.10164],       #img_16-57-49
    [0.078587, -1.1678, 0.15703, 0.97304, 0.11126, 0.1295, -0.15506],       #img_16-58-55
    [0.047077, -0.87793, 0.18785, 0.88671, -0.41326, 0.19021, 0.08231],       #img_17-04-17
    [0.1101, -0.7853, 0.14067, 0.95529, -0.14655, 0.11489, 0.22967],       #img_17-07-21
    [0.33833, -0.84934, 0.12436, 0.9075, -0.33481, -0.119, 0.22403],       #img_17-12-14
    [-0.064699, -0.89433, 0.14303, -0.57633, 0.76346, -0.24827, 0.15273],       #img_17-13-54
    [0.32554, -0.86472, 0.12421, -0.59145, 0.75386, -0.018987, -0.2855],       #img_17-16-30
    [-0.074494, -1.0196, 0.14463, -0.36378, 0.88656, -0.07824, 0.27486],       #img_17-18-27
    [0.20467, -1.035, 0.1665, -0.43361, 0.89982, 0.038556, -0.028459],       #img_17-19-56
]

REnd2Base = []
TEnd2Base = []
for pose in poses:
    # Extract translation and quaternion
    translation = pose[:3]
    quaternion = pose[3:]
    
    # Convert quaternion to rotation matrix
    REnd2Base.append(R.from_quat(quaternion).as_matrix())
    TEnd2Base.append(translation)

REnd2Base = np.array(REnd2Base)
TEnd2Base = np.array(TEnd2Base)

for i in range(5):
    print("Method:", i)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        REnd2Base,
        TEnd2Base,
        RTarget2Cam,
        TTarget2Cam,
        method=i
    )
    print(R_cam2gripper)
    print(t_cam2gripper)