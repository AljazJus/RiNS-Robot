#!/usr/bin/python3

import rospy
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import String, ColorRGBA, Bool
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import sys
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Quaternion, Twist, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
import math
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler,euler_from_quaternion
from sklearn.cluster import KMeans
import pickle



class cylinder_inspector():
    def __init__(self):

        rospy.init_node('cylinder_inspector', anonymous=True)
        
        self.park_sub = rospy.Subscriber("inspect_initiated", Bool, self.inspect_callback)
        
        # Set up the action client for the move_base action
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        
        self.arm_movement_pub = rospy.Publisher('/turtlebot_arm/arm_controller/command', JointTrajectory, queue_size=1)
        self.arm_user_command_sub = rospy.Subscriber("/arm_command", String, self.new_user_command)
        self.markers_pub = rospy.Publisher('park_points', Marker, queue_size=100)

        # Just for controlling wheter to set the new arm position
        self.user_command = None
        self.send_command = False

        # Pre-defined positions for the arm
        self.retract = JointTrajectory()
        self.retract.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.retract.points = [JointTrajectoryPoint(positions=[0,-1.3,2.2,1],
                                                    time_from_start = rospy.Duration(1))]

        self.extend = JointTrajectory()
        self.extend.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.extend.points = [JointTrajectoryPoint(positions=[0,0.2,0.4,0.3],
                                                    time_from_start = rospy.Duration(1))]
        
        # new JointTrajectoryPoint(positions=[0,1,0.5,0.0],
        # old JointTrajectoryPoint(positions=[0,0.1,1,0.3]
        self.right = JointTrajectory()
        self.right.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.right.points = [JointTrajectoryPoint(positions=[-2.5,1.2,0.2,0.0],
                                                    time_from_start = rospy.Duration(4))]
        
        self.left = JointTrajectory()
        self.left.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.left.points = [JointTrajectoryPoint(positions=[2.5,1.2,0.2,0.0],
                                                    time_from_start = rospy.Duration(6))]
        
        self.check = JointTrajectory()
        self.check.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.check.points = [JointTrajectoryPoint(positions=[0,0.4,0.6,0.5],
                                                    time_from_start = rospy.Duration(1))]

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)
        
        self.window_dim = 3
        self.epsilon = 0.01
        self.colors = ( (np.array([0.22023449, 0.95504346, 0.19155043]), 'green'),
                        (np.array([0.18645276, 0.18645276, 0.18645276]), 'black'),
                        (np.array([0.36964507, 0.6392554, 0.97609829]), 'blue'),
                        (np.array([0.47018454, 0.23760092, 0.22946943]), 'red'))
        
        self.min_limit = 0.05

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1
        
        self.completion_pub = rospy.Publisher('/prisoner_jail', Bool, queue_size=1)

        # Initialize the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier('/home/edin/Desktop/ROS/src/exercise4/scripts/haarcascade_frontalface_default.xml')

        # Initialize the face recognizer
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        self.face_recognizer.read('/home/edin/Desktop/ROS/src/exercise4/recognition_files/face_dataset.xml')

        # Initialize the face labels
        with open('/home/edin/Desktop/ROS/src/exercise4/recognition_files/face_labels.pkl', 'rb') as f:
            self.face_labels= pickle.load(f)

        self.face_labels = {v: k for k, v in self.face_labels.items()}
        print(self.face_labels)
        rospy.sleep(0.5)
        self.arm_movement_pub.publish(self.extend)
        rospy.sleep(0.5)
        

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        rospy.loginfo("Cylinder inspector initialized")

        # call park function until we are done
    def nearest_neighbour(self, col):
        min_dist = 100
        best_index = -1
        for i in range(len(self.colors)):
            dist = np.sum((self.colors[i][0] - col) ** 2)
            if dist < min_dist:
                best_index = i
                min_dist = dist
        return self.colors[best_index][1]
    
    def inspect_callback(self, msg):
        if msg.data:
            # Subscribe to the image and/or depth topic
            self.image_sub = rospy.Subscriber("/arm_camera/rgb/image_raw", Image, self.image_callback)
            self.depth_sub = rospy.Subscriber("/arm_camera/depth_registered/image_raw", Image, self.depth_callback)
       
    def image_callback(self, data):
            
        print('I got a new image!')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = cv_image.shape

        # Tranform image to gayscale and blur
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Detect faces in the image
        faces = self.recognize_faces(gray)

        cv2.imshow("Image window", faces)
        cv2.waitKey(1)
        try:
            depth_img_msg = rospy.wait_for_message('/arm_camera/depth/image_raw', Image)

        except Exception as e:
            print(e)
            
    def depth_callback(self,data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            print(e)

        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 =image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)
        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)
    
        
    def new_user_command(self, data):
        self.user_command = data.data.strip()
        self.send_command = True

    def update_position(self):
        # Only if we had a new command
        if self.send_command:
            if self.user_command == 'retract':
                self.arm_movement_pub.publish(self.retract)
                print('Retracted arm!')
            elif self.user_command == 'extend':
                self.arm_movement_pub.publish(self.extend)
                print('Extended arm!')
            elif self.user_command == 'right':
                self.arm_movement_pub.publish(self.right)
                print('Right-ed arm!')
            else:
                print('Unknown instruction:', self.user_command)
                return(-1)
            self.send_command = False

    def recognize_faces(self, image):
        ### recognize face and return bounty
        
def main():

    am = cylinder_inspector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

