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

    # mtx = np.array([[2247.24, 0., 1019.52],
    #                 [0., 2245.15, 781.512],
    #                 [0., 0., 1.]])
    # dist = np.array([[-0.127476, 0.157168, 0.000106415, -0.000806452, -0.04221411]]) 

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
image_folder = "./2025-02-04"
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
    [0.6839892720522448, 0.1976482153951885, 1.1680585031908775, 0.7105275851498413, 0.0454139094868589, -0.6414416152482193, 0.2857285106425499],
    [0.6850679759956944, 0.1892168127486337, 1.1714483326775693, 0.6278294358691929, 0.1358992131842169, -0.6241628347984302, 0.4447272860624467],
    [0.6761103612095217, 0.1808447125617935, 1.1740707883579227, 0.7161985406410009, -0.1566767367773201, -0.6540226623502423, 0.1864575224248461],
    [0.6886379185583256, 0.1894786682745075, 1.1721615451564968, -0.0232772568783367, 0.5818038734001152, -0.2274327036110555, 0.7805362179534070],
    [0.5454103042699247, -0.1009055062426236, 1.1200747901650117, -0.0896306880503133, 0.5087467168945294, 0.1290300363871505, 0.8464599030778691],
    [0.5430830685424871, -0.1047200315613783, 1.1185259374764405, 0.5219928626374143, 0.4513360959092531, -0.2874631275263941, 0.6642169300746068],
    [0.5371590399388966, -0.1034864169738027, 1.1160757714692586, 0.6806069051781645, 0.3680739460910680, -0.4250122295658960, 0.4697450537814920],
    [0.5447135927281065, -0.1084042319486798, 1.1191448607897496, 0.7838430561376980, 0.2320994843176093, -0.4737256031729524, 0.3275728096509780],
    [0.6318727972414468, 0.1940646433639928, 1.1501185350944225, 0.0095740920320168, 0.5509820355441590, -0.2088611391840107, 0.8079010816975893],
    [0.6366949003163759, 0.1966315541195149, 1.1509093275387170, -0.3404664363742495, 0.5656225400609461, 0.0559259291852392, 0.7490167143144337],
    [0.6259049802971872, 0.1970577979205853, 1.1472289319204851, -0.2248057182038017, 0.6138009531343597, -0.0134115953321014, 0.7566577218960185],
    [0.6281458303814456, 0.1940193385870780, 1.1488784442304862, 0.6917160504012978, -0.4604770037842728, -0.3921769462193697, -0.3945720180868604],
    [0.9587549752760265, -0.2368856015418742, 1.1172276687811378, 0.4721052737937634, 0.6353271067309102, -0.4113110139648520, 0.4519948314972770],
    [0.9551371483668982, -0.2278945061182328, 1.1204902467427276, 0.0582528380005915, 0.7467556557524989, 0.1812682044762386, 0.6372632387895075],
    [0.9537540669909357, -0.2228318991203667, 1.1225129849956816, -0.0651050338450270, 0.6829457326059771, 0.3621442604143034, 0.6310293143208962],
    [0.9417268369014761, -0.2333354312418214, 1.1223001405925053, 0.4536802915147383, 0.6665899177014480, -0.4614645059273271, 0.3699764647638100],
    [0.9240404999424502, -0.0378395853425545, 1.1687252250953004, -0.0050662818996347, 0.7670044623577972, 0.0340663648070661, 0.6407167629302137],
    [0.9242456632767913, -0.0384004001379934, 1.1686127432584397, 0.2054393899877298, 0.7338898527234691, -0.2232118918051327, 0.6077637637008053],
    [0.9163051431566670, -0.0487300319984039, 1.1705426157817493, 0.3259330008511366, 0.6363926651525449, -0.4720098561846114, 0.5157312772760599],
    [0.9186349060931503, -0.0407203034554954, 1.1698436040191991, -0.5422059025928760, -0.3860501507580347, 0.6834211308877485, -0.2998559623373431],
    [0.9046813529111839, 0.1398732581709072, 1.1568713841763285, -0.1721629710534618, 0.7304848686046777, -0.0790080534288970, 0.6561322241980105],
    [0.8955009311886423, 0.1463333509316608, 1.1558133280508880, 0.1162506204042143, 0.6780921064018098, -0.3546842652177697, 0.6331476608960366],
    [0.9068267688252907, 0.1415422752675189, 1.1559725677551007, 0.2842929031989655, 0.5210214541252479, -0.5079669423262172, 0.6242465658981788],
    [0.8896061711341349, 0.1514283038226950, 1.1555574478440789, -0.4517472467893878, -0.3803005263599455, 0.6857797295782934, -0.4254434124158926]

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