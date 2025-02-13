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
    [0.6839684151116203, 0.1975366648065884, 1.1681678308346723, 0.7105717366678795, 0.0452974867698955, -0.6414066664904633, 0.2857156504686371],
    [0.6850811089499939, 0.1892927954541967, 1.1714180767555378, 0.6277993787950157, 0.1359211560520387, -0.6242386818842987, 0.4446565498915127],
    [0.6760656206237601, 0.1807602386415525, 1.1740457729878331, 0.7162212168967544, -0.1566294181105392, -0.6540256723556036, 0.1863996076956357],
    [0.6885991574142589, 0.1895611262497038, 1.1721620564001287, -0.0233818386661810, 0.5817988836277783, -0.2273838613290008, 0.7805510414041951],
    [0.5453903548572963, -0.1009170948852092, 1.1201481028952858, -0.0896258575121352, 0.5087469112405535, 0.1290086253077692, 0.8464635612739692],
    [0.5431092393228147, -0.1047928165385194, 1.1184824025645037, 0.5220592314085932, 0.4513188498748983, -0.2875058844436548, 0.6641579789921520],
    [0.5371162303816682, -0.1034815008897453, 1.1161329140863820, 0.6805803632578369, 0.3680374589129611, -0.4249864314672144, 0.4698354297554455],
    [0.5447848966470289, -0.1083775034814171, 1.1189517329474730, 0.7837604928935205, 0.2322184052976091, -0.4737971062811744, 0.3275826675816761],
    [0.6318630563053343, 0.1940455767289496, 1.1502210240866813, 0.0096385329808082, 0.5509559928992995, -0.2088906519051280, 0.8079104456045704],
    [0.6368234137223362, 0.1965678316086742, 1.1509836062034313, -0.3403644517443397, 0.5656698675817031, 0.0558537162909155, 0.7490327117525425],
    [0.6258272833092533, 0.1971621603341699, 1.1471403846572192, -0.2247793099073575, 0.6138308556211727, -0.0133250724532066, 0.7566428384443138],
    [0.6280733713262564, 0.1939768505290065, 1.1488705078616517, 0.6917408724638336, -0.4604694488385639, -0.3921225234529316, -0.3945914072178822],
    [0.9587475382938341, -0.2370686747755036, 1.1173308208851387, 0.4721825512978131, 0.6352866949017759, -0.4113085600272432, 0.4519731429841621],
    [0.9552370131932999, -0.2279421274642859, 1.1203820919900063, 0.0582862630448582, 0.7467561627163513, 0.1812226766335066, 0.6372725370351398],
    [0.9537636436625189, -0.2228285853592450, 1.1226221892353347, -0.0650788297885600, 0.6829269857512046, 0.3621034972868257, 0.6310756969641406],
    [0.9416516351315213, -0.2333515017189901, 1.1221883391970457, 0.4536336906121295, 0.6666036730675273, -0.4615095732810198, 0.3699526071870164],
    [0.9240524531986927, -0.0379003589816982, 1.1688307922139738, -0.0051255196315570, 0.7669502021966211, 0.0340494010135064, 0.6407821429235446],
    [0.9243202515268651, -0.0385166502443145, 1.1685556990858417, 0.2054533808669739, 0.7338970818446030, -0.2232641434750247, 0.6077311114207252],
    [0.9163503595829736, -0.0488475565397128, 1.1704830250346632, 0.3260000629281647, 0.6363721306900839, -0.4720308437569898, 0.5156950191672012],
    [0.9186379039392629, -0.0405797128411017, 1.1696220328517584, -0.5420870358486427, -0.3861841694399603, 0.6834539779421435, -0.2998234361651214],
    [0.9047582006992215, 0.1398397974181928, 1.1568795104879452, -0.1722229031815187, 0.7304789783851569, -0.0790103572815001, 0.6561227760102195],
    [0.8954823875738922, 0.1462497927933994, 1.1557482318130110, 0.1163330911161965, 0.6781152815656737, -0.3547871577107992, 0.6330500371550026],
    [0.9068302055502517, 0.1414815879909207, 1.1557422015101577, 0.2841403293471649, 0.5210704194783579, -0.5080477496711950, 0.6242094001512373],
    [0.8895872536988985, 0.1514348254247163, 1.1553749838222385, -0.4516778913239908, -0.3803647128356379, 0.6858360203928792, -0.4253689232304518]

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