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
image_folder = "./2025-02-03"
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
RTarget2Cam, TTarget2Cam = compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, True)

# Poses with translation and quaternion data
# Each pose is now: [tx, ty, tz, qx, qy, qz, qw]
poses = [
    # pc capture poses
    [0.6807048889083127, 0.1880334876234044, 1.1712027763450696, 0.3656767003079450, 0.3981874452663681, -0.5205546093838458, 0.6608707951887178],          
    [0.5420846353351662, -0.1138629027562370, 1.1177544524750487, 0.0315939494922597, 0.5399611968137666, -0.0205956791102804, 0.8408445434757317],       
    # image capture poses
    [0.64912, 0.1122, 1.1581, 0.39749, 0.42905, -0.53305, 0.61138],       
    [0.53277, 0.071117, 1.1413, 0.24824, 0.48225, -0.27546, 0.79368],       
    [0.64022, 0.076462, 1.1107, 0.35587, 0.5084, -0.44457, 0.64595],     
    [0.83542, 0.096346, 1.1047, 0.28899, 0.56044, -0.49295, 0.59949],  
    [0.90074, 0.094811, 1.1667, -0.40198, -0.41984, 0.68598, -0.43769],
    [0.82849, 0.050274, 1.1242, -0.46862, -0.45267, 0.58549, -0.48238],  
    [0.83373, -0.20885, 1.1176, 0.49063, 0.59141, -0.49032, 0.41122],  
    [0.8157, -0.26575, 1.0385, 0.53496, 0.61902, -0.42688, 0.38523],           
    [0.85075, -0.2189, 1.1162, 0.41456, 0.68047, -0.38802, 0.46318],  
    [0.75787, -0.35674, 1.0241, 0.58759, 0.64324, -0.35232, 0.34185],       
    [0.48029, 0.01379, 1.0632, 0.7983, 0.00070485, -0.60224, -0.005433],    
    [0.5905, -0.18101, 1.1093, 0.70648, 0.34972, -0.55405, 0.26762], 
    [0.62166, -0.201, 1.0512, 0.62897, 0.48336, -0.4877, 0.36456],       
    [0.94054, -0.14377, 1.0131, 0.26012, 0.79349, -0.26172, 0.48395],
    [0.73209, -0.065094, 1.0368, 0.25879, 0.65841, -0.31371, 0.63333],
    [0.78742, -0.17075, 1.0948, 0.49753, 0.53135, -0.46908, 0.5001],
    [0.89824, -0.042257, 1.0703, 0.032616, 0.82246, -0.071386, 0.56338],
    [0.61143, 0.023266, 1.093, 0.45191, 0.49243, -0.52031, 0.53158]
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