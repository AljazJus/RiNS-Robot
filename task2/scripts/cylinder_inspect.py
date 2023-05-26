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

from task2.srv import CylinderInspect, CylinderInspectResponse



class cylinder_inspector():
    def __init__(self):

        rospy.init_node('cylinder_inspector', anonymous=True)
        
        self.park_sub = rospy.Service("initiate_inspect", CylinderInspect, self.inspect_callback)
        
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
        self.right.points = [JointTrajectoryPoint(positions=[-2.5,-0.2,0.4,0.8],
                                                    time_from_start = rospy.Duration(4))]
        
        self.left = JointTrajectory()
        self.left.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.left.points = [JointTrajectoryPoint(positions=[2.5,-0.2,0.4,0.8],
                                                    time_from_start = rospy.Duration(6))]
        

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
        

        self.face_detected = False
        self.turns_completed = 0

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
        # Subscribe to the image and/or depth topic
        
        try:
            img = rospy.wait_for_message("/arm_camera/rgb/image_raw", Image)
            print("Got a new image!")
        except Exception as e:
            print(e)
            return 0
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)


        self.dims = cv_image.shape

        # Tranform image to gayscale and blur
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        val = 0

        
        right = self.right.points[0].positions
        left = self.left.points[0].positions
        extend = self.extend.points[0].positions

        while(self.turns_completed < 2):
            result = rospy.wait_for_message('/turtlebot_arm/arm_controller/state', JointTrajectoryControllerState)
            actual = result.actual.positions
            if abs(actual[0] - extend[0]) < 0.05 and abs(actual[1] - extend[1]) < 0.05 and abs(actual[2] - extend[2]) < 0.05 and abs(actual[3] - extend[3]) < 0.5:
                self.arm_movement_pub.publish(self.right)
                rospy.loginfo('Right-ed arm!')
                print("Now should turn left")
            elif abs(actual[0] - right[0]) < 0.05 and abs(actual[1] - right[1]) < 0.05 and abs(actual[2] - right[2]) < 0.05 and abs(actual[3] - right[3]) < 0.5:
                self.arm_movement_pub.publish(self.left)
                rospy.loginfo('Left-ed arm!')
            elif abs(actual[0] - left[0]) < 0.05 and abs(actual[1] - left[1]) < 0.05 and abs(actual[2] - left[2]) < 0.05 and abs(actual[3] - left[3]) < 0.5:
                self.arm_movement_pub.publish(self.extend)
                rospy.loginfo('Extend-ed arm!')
                rospy.sleep(1)
                self.turns_completed += 1
        else:
            self.arm_movement_pub.publish(self.retract)

        return val
        
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
        

            
        
def main():

    am = cylinder_inspector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

