import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd

def find_chessboard_corners(images, pattern_size, ShowCorners=False):
        """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
        chessboard_corners = []
        IndexWithImg = []
        i = 0
        print("Finding corners...")
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if ret:
                chessboard_corners.append(corners)

                cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                if ShowCorners:
                    #plot image using maplotlib. The title should "Detected corner in image: " + i
                    plt.imshow(image)
                    plt.title("Detected corner in image: " + str(i))
                    plt.show()
                #Save the image in a folder Named "DetectedCorners"
                #make folder
                # if not os.path.exists("DetectedCorners"):
                #     os.makedirs("DetectedCorners")

                # cv2.imwrite("DetectedCorners/DetectedCorners" + str(i) + ".png", image)

                IndexWithImg.append(i)
                i = i + 1
            else:
                print("No chessboard found in image: ", i)
                i = i + 1
        return chessboard_corners, IndexWithImg


def calculate_intrinsics(chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
        """Calculates the intrinc camera parameters fx, fy, cx, cy from the images"""
        # Find the corners of the chessboard in the image
        imgpoints = chessboard_corners
        # Find the corners of the chessboard in the real world
        objpoints = []
        for i in range(len(IndexWithImg)):
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp)
        # Find the intrinsic matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)

        print("The projection error from the calibration is: ", calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowProjectError))
        return mtx


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowPlot=False):
        """Calculates the reprojection error of the camera for each image. The output is the mean reprojection error
        If ShowPlot is True, it will show the reprojection error for each image in a bar graph"""

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

        # if ShowPlot:
        #     # Plotting the bar graph
        #     fig, ax = plt.subplots()
        #     img_indices = range(1, len(errors) + 1)
        #     ax.bar(img_indices, errors)
        #     ax.set_xlabel('Image Index')
        #     ax.set_ylabel('Reprojection Error')
        #     ax.set_title('Reprojection Error for Each Image')
        #     plt.show()
        #     print(errors)

        #     #Save the bar plot as a .png
        #     fig.savefig('ReprojectionError.png')

        return mean_error



def compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, Testing=False):
        """Takes the chessboard corners and computes the camera poses"""
        # Create the object points.Object points are points in the real world that we want to find the pose of.
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        for corners in chessboard_corners:
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)
            # rvec is the rotation vector, tvec is the translation vector
            if Testing == True:
                print("Current iteration: ", i, " out of ", len(chessboard_corners[0]), " iterations.")

                # Convert the rotation vector to a rotation matrix
                print("rvec: ", rvec)
                print("rvec[0]: ", rvec[0])
                print("rvec[1]: ", rvec[1])
                print("rvec[2]: ", rvec[2])
                print("--------------------")
            i = 1 + i
            R, _ = cv2.Rodrigues(rvec)  # R is the rotation matrix from the target frame to the camera frame
            RTarget2Cam.append(R)
            TTarget2Cam.append(tvec)

        return RTarget2Cam, TTarget2Cam





image_folder = "T:/Pose_Handheld/NDI_test/third/calibration"
image_files = sorted(glob.glob(f'{image_folder}/*.png'))
images = [cv2.imread(f) for f in image_files]

pattern_size=(9, 8)
square_size=20/1000
ShowProjectError=True
ShowCorners=False

# camera calibration
chessboard_corners, IndexWithImg = find_chessboard_corners(images, pattern_size, ShowCorners=ShowCorners)
intrinsic_matrix = calculate_intrinsics(chessboard_corners, IndexWithImg,
                                        pattern_size, square_size,
                                        images[0].shape[:2], ShowProjectError = ShowProjectError)


#Calculate camera extrinsics
RTarget2Cam, TTarget2Cam = compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix)




pose_file = "T:/Pose_Handheld/NDI_test/third/calibration/JointStateSteps.csv"

poses = []

df = (pd.read_csv(pose_file).values)

for i in range(0, len(df)):
    pose = df[i][0].split(',')
    pose_float = []
    for value in pose:
        pose_float.append(float(value))
    poses.append(pose_float)





# with open(pose_file, "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')
#         line = line.replace(" ", "")
#         if line[:7] == "link_6:":
#             pose = line[7:].split(';')
#             pose_float = []
#             for value in pose:
#                 pose_float.append(float(value))

#             poses.append(pose_float)


REnd2Base = []
TEnd2Base = []
for pose in poses:
    orn = pose[-4:]
    REnd2Base.append(R.from_quat(orn).as_matrix())
    TEnd2Base.append(pose[:3])

REnd2Base = np.array(REnd2Base)
TEnd2Base = np.array(TEnd2Base)

for i in range(0, 5):
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
    a = 1
    

