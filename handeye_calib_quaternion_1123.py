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
        if ret:  # Only process if chessboard corners are found
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
            chessboard_corners.append(corners)

            cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            if ShowCorners:
                plt.imshow(image)
                plt.title("Detected corner in image: " + str(i))
                plt.show()

            IndexWithImg.append(i)
        else:
            print(f"No chessboard found in image: {i}")  # Log failed images
        i += 1
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
image_folder = "./2024-11-23"
image_files = sorted(glob.glob(f'{image_folder}/img_*.png'))
images = [cv2.imread(f) for f in image_files]

pattern_size = (6, 5)
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
    [0.10076220340496435, -0.8350636088670056, 0.5399425293099728, 0.031269994456966745, 0.19279098503458428, -0.8156695974368794, 0.08361089979792864],
    [0.007476937345067267, 0.8464301018076564, -0.5196154224902066, 0.11618946110617055, 0.14660590502003168, -0.6846951092040476, 0.11193899222691063],
    [0.07646891036003954, -0.7634896561286731, 0.6091946925109409, -0.20029447658227803, 0.08186171437311554, -0.7032758344374039, 0.10466125204977654],
    [0.22099064396060283, -0.7952202682847488, 0.5622369261800323, -0.05174455556519199, 0.07953020438655439, -0.8208629112465771, 0.11079481179110609],
    [0.20213027961420343, -0.9244292141462087, 0.3066630317197714, -0.1026243785488879, 0.057897228635413314, -0.844888291921343, 0.09147076659922324],
    #[0.17070213633925732, -0.9837686932942945, -0.03312349262808584, 0.044303193782922065, 0.20864639633362328, -0.8705851845136263, 0.08992303598567848],
    [0.10430096581266957, -0.9701342212393357, 0.1625595117521107, 0.14674912759470676, 0.2834018982812924, -0.8117824227335609, 0.1404760130460035],
    [0.011442525720843436, -0.9853715166756595, 0.12424293842527682, 0.11608503340608256, 0.33928685304580003, -0.7710485638567551, 0.14047475799461495],
    [0.06943733209283777, 0.9618817016066857, -0.22612570864107998, -0.13721957920655914, 0.3541974984245776, -0.7170846364820378, 0.1404767948023691],
    [0.13537920692828592, 0.9663998651254534, -0.20414686411279545, -0.07789627006401291, 0.3273915088298732, -0.666146381361912, 0.1256534907541716],
    [0.18671284034231014, 0.9778413378046376, -0.09431998835924565, 0.008268804487675151, 0.23332606369842637, -0.6232109583101895, 0.1307124201101345],
    [0.18013800703600433, 0.9789546661503143, 0.0127228070742904, 0.09505887766943999, 0.09627418026056929, -0.6551873253971212, 0.11255953812430712],
    [0.14676761797383311, 0.9754228190509249, 0.13327129010516933, 0.09616836081964543, 0.10614736732669161, -0.6448414244503367, 0.11861159055387027],
    [0.13140533833833265, 0.9462129718885854, 0.2950695549073549, -0.018644212311998976, 0.22102356194195782, -0.6689001796185841, 0.12787211057573467],
    [0.09402660181663906, 0.814579224704971, 0.5504233100944989, -0.15701549138374457, 0.2862942241534878, -0.6242989780623049, 0.12774821674634446],
    [0.19213714553359384, 0.6464898570250361, 0.7285783940253824, -0.1196148227827379, 0.1662478868121979, -0.565777284513841, 0.10432536246006884],
    [0.12555534034009638, 0.6790243368163188, 0.7215621418216156, -0.0500987226687842, 0.15057724898813352, -0.6997060800674487, 0.10432513239476941],
    [0.04402906915807106, 0.49710462135500655, 0.8663641309376184, -0.01901654861625843, 0.20648011459476381, -0.7224662296958205, 0.15626125643903593],
    [0.05253538504294503, 0.728033029189299, 0.6816269340810345, 0.05091821347152542, 0.18433931115941904, -0.7533615663646885, 0.12439437985579006],
    [0.049196446581164344, -0.5303018826975658, -0.8447508981123596, -0.05249326613543675, 0.22298909900941769, -0.796716217731753, 0.12439994078329443]
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