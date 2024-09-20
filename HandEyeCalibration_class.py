import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

import csv
from scipy.spatial.transform import Rotation as Rot

class CameraCalibration:
    """Camera calibration class, this class takes as input a folder with images and a folder with the corresponding Base2endeffector transforms
    and outputs the intrinsic matrix in a .npz file. It also performs hand-eye calibration and saves those results in a .npz file.
    The images with the corner detection are saved in a folder called 'DetectedCorners'

    This class has 4 optional parameters:
    pattern_size: the number of corners in the chessboard pattern, default is (4,7)
    square_size: the size of the squares in the chessboard pattern, default is 33/1000
    ShowProjectError: if True, it will show the reprojection error for each image in a bar plot, default is False
    ShowCorners: if True, it will show the chessboard corners for each image, default is False

    """
    def __init__(self, image_folder, pattern_size=(4, 7), square_size=33/1000, ShowProjectError=False, ShowCorners=False):

        #Initiate parameters
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.objpoints, self.imgpoints, self.rvecs, self.tvecs, self.intrinsic_matrix, self.dist_coeffs = [], [], [], [], [], []

        #load images and joint positions
        self.image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        # self.transform_files = sorted(glob.glob(f'{Transforms_folder}/*.npz'))
        self.images = [cv2.imread(f) for f in self.image_files]
        self.images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in self.images]

        # self.All_T_base2EE_list = [np.load(f)['arr_0'] for f in self.transform_files]
        end_to_base_quat = []
        with open(image_folder + "JointStateSteps.csv", 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                if 'marker' in row:
                    continue
                row_list = [float(x) for x in row[0].split(',')]
                end_to_base_quat.append(row_list) 
        end_to_base_quat = np.array(end_to_base_quat)
        self.All_T_base2EE_list = end_to_base_quat

        #find chessboard corners and index of images with chessboard corners
        self.chessboard_corners, self.IndexWithImg = self.find_chessboard_corners(self.images, self.pattern_size, ShowCorners=ShowCorners)
        self.intrinsic_matrix = self.calculate_intrinsics(self.chessboard_corners, self.IndexWithImg,
                                                           self.pattern_size, self.square_size,
                                                           self.images[0].shape[:2], ShowProjectError = ShowProjectError)

        #Remove transforms were corners weren't detected
        self.T_base2EE_list = [self.All_T_base2EE_list[i] for i in self.IndexWithImg]

        #save intrinsic matrix
        np.savez("IntrinsicMatrix.npz", self.intrinsic_matrix)
        #Calculate camera extrinsics
        self.RTarget2Cam, self.TTarget2Cam = self.compute_camera_poses(self.chessboard_corners,
                                                                       self.pattern_size, self.square_size,
                                                                       self.intrinsic_matrix)

        #Convert to homogeneous transformation matrix
        self.T_target2cam = [np.concatenate((R, T), axis=1) for R, T in zip(self.RTarget2Cam, self.TTarget2Cam)]
        for i in range(len(self.T_target2cam)):
            self.T_target2cam[i] = np.concatenate((self.T_target2cam[i], np.array([[0, 0, 0, 1]])), axis=0)

        #Calculate T_cam2target
        self.T_cam2target = [np.linalg.inv(T) for T in self.T_target2cam]
        self.R_cam2target = [T[:3, :3] for T in self.T_cam2target]
        self.R_vec_cam2target = [cv2.Rodrigues(R)[0] for R in self.R_cam2target]
        self.T_cam2target = [T[:3, 3] for T in self.T_cam2target]   #4x4 transformation matrix
        
        #Calculate T_Base2EE
        self.REE2Base, self.tEE2Base = [], []
        self.Rbase2ee, self.Tbase2ee = [], []
        for i in range(len(self.T_base2EE_list)):
            rot = Rot.from_quat(self.T_base2EE_list[i][3:]).as_matrix() #quat scalar-last (x, y, z, w) format          
            self.Rbase2ee.append(rot)            
            self.Tbase2ee.append(self.T_base2EE_list[i][0:3])

            homo_matrix = np.eye(4)
            homo_matrix[:3, :3] = rot
            homo_matrix[:3, 3] = self.T_base2EE_list[i][0:3]
            inv_homo_matrix = np.linalg.inv(homo_matrix)
            self.REE2Base.append(inv_homo_matrix[:3, :3])
            self.tEE2Base.append(inv_homo_matrix[:3, 3].reshape((3,1)))

        #Create folder to save final transforms
        if not os.path.exists("FinalTransforms"):
            os.mkdir("FinalTransforms")

        methods = [
            ('Tsai-Lenz', cv2.CALIB_HAND_EYE_TSAI),
            ('Park', cv2.CALIB_HAND_EYE_PARK),
            ('Horaud', cv2.CALIB_HAND_EYE_HORAUD),
            ('Andreff', cv2.CALIB_HAND_EYE_ANDREFF),
            ('Daniilidis', cv2.CALIB_HAND_EYE_DANIILIDIS)
        ]

        best_method = None
        # unbounded upper value for comparison. Useful for finding lowest values
        best_rms_error = float('inf')
        #solve hand-eye calibration
        for method_name, method in methods:
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(
                self.Rbase2ee,
                self.Tbase2ee,
                self.RTarget2Cam,
                self.TTarget2Cam,
                method=method
            )
            #print and save each results as .npz file
            print(f"The Results for Method {method_name}:")
            print("R_cam2gripper:", self.R_cam2gripper)
            print("t_cam2gripper:", self.t_cam2gripper)
            
            # eef2cam_matrix = np.eye(4)
            # eef2cam_matrix[:3, :3] = self.R_cam2gripper
            # eef2cam_matrix[:3, 3] = self.t_cam2gripper.reshape(3)
            # cam2ee_matrix = np.linalg.inv(eef2cam_matrix)
            # 2 means transform to, i.e. the 1st frame is the base frame
            # print("eef2cam_matrix:\n", eef2cam_matrix)
            # print("cam2ee_matrix:\n", cam2ee_matrix)
            
            #Create 4x4 transfromation matrix
            self.T_cam2gripper = np.concatenate((self.R_cam2gripper, self.t_cam2gripper), axis=1)
            self.T_cam2gripper = np.concatenate((self.T_cam2gripper, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_cam2gripper_Method_{i}.npz", self.T_cam2gripper)
            #Save the inverse transfrom too
            self.T_gripper2cam = np.linalg.inv(self.T_cam2gripper)
            np.savez(f"FinalTransforms/T_gripper2cam_Method_{i}.npz", self.T_gripper2cam)

            # Recalculate reprojection error with this hand-eye calibration result
            # _, _, rms_error = self.calculate_projection_error(self.objpoints, self.imgpoints, self.rvecs, self.tvecs, self.intrinsic_matrix, self.dist_coeffs)
            # print(f"Method: {method_name}, RMS Reprojection Error: {rms_error}")
            # if rms_error < best_rms_error:
            #     best_rms_error = rms_error
            #     best_method = method_name        
        
        #solve hand-eye calibration using calibrateRobotWorldHandEye
        robot_world_methods = [
            ('Shan', cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
            ('Li', cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
        ]
        for robot_world_method_name, robot_world_method in robot_world_methods:
            self.R_base2world, self.t_base2world, self.R_gripper2cam, self.t_gripper2cam= cv2.calibrateRobotWorldHandEye(
                self.RTarget2Cam,
                self.TTarget2Cam,
                self.REE2Base,
                self.tEE2Base,
                method=robot_world_method)
            #print and save each results as .npz file
            print(f"The Results for Method calibrateRobotWorldHandEye {robot_world_method_name}:")
            print("R_cam2gripper:", self.R_gripper2cam)
            print("t_cam2gripper:", self.t_gripper2cam)
            #Create 4x4 transfromation matrix T_gripper2cam
            self.T_gripper2cam = np.concatenate((self.R_gripper2cam, self.t_gripper2cam), axis=1)
            self.T_gripper2cam = np.concatenate((self.T_gripper2cam, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_gripper2cam_Method_{i+4}.npz", self.T_gripper2cam)
            #save inverse too
            self.T_cam2gripper = np.linalg.inv(self.T_gripper2cam)
            np.savez(f"FinalTransforms/T_cam2gripper_Method_{i+4}.npz", self.T_cam2gripper)

            # Recalculate reprojection error with this hand-eye calibration result
            # _, _, rms_error = self.calculate_projection_error(self.objpoints, self.imgpoints, self.rvecs, self.tvecs, self.intrinsic_matrix, self.dist_coeffs)
            # print(f"Method: {robot_world_method_name}, RMS Reprojection Error: {rms_error}")
            # if rms_error < best_rms_error:
            #     best_rms_error = rms_error
            #     best_method = robot_world_method_name

        # print(f"Best Hand-Eye Calibration Method: {best_method} with RMS Error: {best_rms_error}")

    def find_chessboard_corners(self, images, pattern_size, ShowCorners=False):
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
                if not os.path.exists("DetectedCorners"):
                    os.makedirs("DetectedCorners")

                cv2.imwrite("DetectedCorners/DetectedCorners" + str(i) + ".png", image)

                IndexWithImg.append(i)
                i = i + 1
            else:
                print("No chessboard found in image: ", i)
                i = i + 1
        return chessboard_corners, IndexWithImg

    def compute_camera_poses(self, chessboard_corners, pattern_size, square_size, intrinsic_matrix, Testing=False):
        """Takes the chessboard corners and computes the camera poses"""
        # Create the object points.Object points are points in the real world that we want to find the pose of.
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        for corners in chessboard_corners:
            # Best-fit(?) solve PnP method for chessboard scenario :SOLVEPNP_IPPE
            # _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None, None, None, False, cv2.SOLVEPNP_IPPE)
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

    def calculate_intrinsics(self, chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
        """Calculates the intrinc camera parameters fx, fy, cx, cy from the images"""
        # Find the corners of the chessboard in the image
        self.imgpoints = chessboard_corners
        # Find the corners of the chessboard in the real world
        # objpoints = []
        if self.objpoints:
            self.objpoints = []

        for i in range(len(IndexWithImg)):
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            self.objpoints.append(objp)
        # Find the intrinsic matrix
        ret, self.intrinsic_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, ImgSize, None, None)

        mean_normalized_error = self.calculate_reprojection_error(self.objpoints, self.imgpoints, self.rvecs, self.tvecs, self.intrinsic_matrix, self.dist_coeffs, ShowProjectError)
        print(f"The Mean Normalized Reprojection Error: {mean_normalized_error}")

        # #Hardcode TJ's cam intrinsic mat and dist_coeffs
        # self.intrinsic_matrix[0] = [3579.0434570313, 0.0,             1246.4215087891]
        # self.intrinsic_matrix[1] = [0.0,             3578.8725585938, 1037.0089111328]
        # self.intrinsic_matrix[2] = [0.0,             0.0,             1.0]
        # print(f"The hardcoded cam intrinsic: {self.intrinsic_matrix}")
        # #dist_coeffs = [k1, k2, p1, p2, k3]
        # self.dist_coeffs = [-0.0672596246, 0.1255717576, 0.0008782994, -0.0014749423, -0.0597795881]
        # self.dist_coeffs = np.array(self.dist_coeffs)
        # print(f"The hardcoded distortion coefficients: {self.dist_coeffs}")
        
        return self.intrinsic_matrix

    # Utility function to calculate projection error for each image
    def calculate_projection_error(self, objpoints, imgpoints, rvecs, tvecs, intrinsic_matrix, dist_coeffs):
        total_normalized_error = 0
        total_squared_error = 0
        errors = []

        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], intrinsic_matrix, dist_coeffs)
            imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
            error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            errors.append(error)
            total_normalized_error += error
            total_squared_error += error ** 2

        mean_normalized_error = total_normalized_error / len(objpoints)
        mean_squared_error = total_squared_error / len(objpoints)
        rms_error = np.sqrt(mean_squared_error)

        return errors, mean_normalized_error, rms_error

    def calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowPlot=False):
        """Calculates the reprojection error of the camera for each image. The output is the mean reprojection error
        If ShowPlot is True, it will show the reprojection error for each image in a bar graph"""

        errors, mean_normalized_error, rms_error = self.calculate_projection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

        if ShowPlot:
            # Plotting the bar graph
            fig, ax = plt.subplots()
            img_indices = range(1, len(errors) + 1)
            ax.bar(img_indices, errors)
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Reprojection Error')
            ax.set_title('Reprojection Error for Each Image')
            plt.show()
            print(errors)

            #Save the bar plot as a .png
            fig.savefig('NormalizedReprojectionError.png')

        return mean_normalized_error

if __name__== "__main__":
    # Create an instance of the class
    # image_folder = "2024-06-06/"
    # calib = CameraCalibration(image_folder, pattern_size=(8, 9), square_size=0.02, ShowProjectError=False, ShowCorners=False)

    image_folder = "2024-06-17/"
    calib = CameraCalibration(image_folder, pattern_size=(8, 9), square_size=0.02, ShowProjectError=True, ShowCorners=False)