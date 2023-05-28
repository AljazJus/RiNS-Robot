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
        self.right.points = [JointTrajectoryPoint(positions=[-1,-0.4,0.4,0.8],
                                                    time_from_start = rospy.Duration(2))]
        
        self.left = JointTrajectory()
        self.left.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.left.points = [JointTrajectoryPoint(positions=[1,-0.4,0.4,0.8],
                                                    time_from_start = rospy.Duration(4))]
        
        self.check = JointTrajectory()
        self.check.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.check.points = [JointTrajectoryPoint(positions=[0,-0.4,0.4,0.8], time_from_start = rospy.Duration(1))]
        

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
        self.moves_completed = 0

        rospy.sleep(0.5)
        self.arm_movement_pub.publish(self.extend)
        rospy.sleep(0.5)
        
        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=100)

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
        
        val = 0

        right = self.right.points[0].positions
        left = self.left.points[0].positions
        extend = self.extend.points[0].positions
        retract = self.retract.points[0].positions
        check = self.check.points[0].positions

        while(self.moves_completed < 2):
            result = rospy.wait_for_message('/turtlebot_arm/arm_controller/state', JointTrajectoryControllerState)
            actual = result.actual.positions
            
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

            try:
                depth_img_msg = rospy.wait_for_message('/arm_camera/depth/image_raw', Image)

            except Exception as e:
                print(e)


            self.dims = cv_image.shape

            # Tranform image to gayscale and blur
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=0, maxRadius=0)
            if circles is not None:
                candidates = []
                circles = np.round(circles[0, :]).astype("int")
        
                # Use K-means clustering to group circles based on their centers
                kmeans = KMeans(n_clusters=1).fit(circles[:, :2])
                center = np.round(kmeans.cluster_centers_).astype("int")[0]
                radius = np.round(np.mean(circles[:, 2])).astype("int")
                
                circles = [(center[0], center[1], radius)]
                for circle in circles:
                    # Get circle center and radius
                    x, y, r = circle
                    # Calculate bounding box coordinates
                    x1 = int(x - r)
                    x2 = int(x + r)
                    y1 = int(y - r)
                    y2 = int(y + r)


                    depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
                    if depth_img is not None:
                        depth_img_window = depth_img[y1:y2, x1:x2]

                    depth_img_window = depth_img[y1:y2, x1:x2]
                    
                    # Calculate mean depth of cropped depth image
                    
                    depth_mean = np.nanmean(depth_img_window)
                    if(np.isnan(depth_mean)):
                        continue

                    # Crop color image based on bounding box coordinates
                    img_window_color = cv_image[y1:y2, x1:x2]

                    # Compute the RGB color of the cropped color image
                    c = np.mean(img_window_color, axis=(0, 1)) / 255

                    # Save candidate
                    candidates.append((circle, depth_mean, c))
                    
                print("Processing is done! found", len(candidates), "candidates for rings")
                if len(candidates) > 0:
                    print("NO CANIDATES FOUND")
                    
                for c in candidates:
                    circle = c[0]
                    depth_mean = c[1]
                    c = c[2]
                    rospy.sleep(1)
                    # the center of the circle
                    x, y = circle[:2]


                    
                    color_name = self.nearest_neighbour(c)
                    
                    cv2.imshow("Detected circles", cv2.circle(cv_image, (int(x), int(y)), int(r), (0, 255, 0), 2))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows() 
                    
                    # self.stop = JointTrajectory()
                    # self.stop.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
                    # self.stop.points = [JointTrajectoryPoint(positions=[self.actual[0],self.actual[1],self.actual[2],self.actual[3]],
                    #                                             time_from_start = rospy.Duration(1))]
                    
                    # #if abs(self.actual[0] - self.stop.points[0].positions[0]) < 0.05 and abs(self.actual[1] - self.stop.points[0].positions[1]) < 0.05 and abs(self.actual[2] - self.stop.points[0].positions[2]) < 0.05 and abs(self.actual[3] - self.stop.points[0].positions[3]) < 0.5:
                    # self.arm_movement_pub.publish(self.stop)
                    # rospy.sleep(2)
                    self.arm_movement_pub.publish(self.check)
                    self.get_pose((x, y), depth_mean, ColorRGBA(c[0], c[1], c[2], 1), color_name)
                    self.moves_completed += 1
                    
                    self.arm_movement_pub.publish(self.extend)

                    break

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
            elif abs(actual[0] - retract[0]) < 0.05 and abs(actual[1] - retract[1]) < 0.05 and abs(actual[2] - retract[2]) < 0.05 and abs(actual[3] - retract[3]) < 0.5:
                self.arm_movement_pub.publish(self.extend)
                rospy.loginfo('Retract-ed arm!')
                rospy.sleep(1)
            elif abs(actual[0] - check[0]) < 0.05 and abs(actual[1] - check[1]) < 0.05 and abs(actual[2] - check[2]) < 0.05 and abs(actual[3] - check[3]) < 0.5:
                self.arm_movement_pub.publish(self.extend)
                rospy.loginfo('Extend-ed arm!')
                rospy.sleep(1)
                
        else:
            self.arm_movement_pub.publish(self.retract)
                    
                    
        
        self.arm_movement_pub.publish(self.check)

        self.moves_completed = 0
        self.turns_completed = 0
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
            
    def get_pose(self,e,dist, marker_color, color_name):
        rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        rospy.loginfo("ENTERED GET POSE FUNCTION WITH DISTANCE OF %f", dist)
        rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Calculate the position of the detected ellipse

        if dist < 0.4:
            return
        
        k_f = 525 # kinect focal length in pixels

        elipse_x = self.dims[1] / 2 - e[0]
        elipse_y = self.dims[0] / 2 - e[1]

        angle_to_target_x = np.arctan2(elipse_x,k_f)
        angle_to_target_y = np.arctan2(-elipse_y,k_f)

        # Get the angles in the base_link relative coordinate system
        x,y,z = dist*np.cos(angle_to_target_x), dist*np.sin(angle_to_target_x), dist * np.sin(angle_to_target_y)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = z
        point_s.point.z = x
        point_s.header.frame_id = "arm_camera_rgb_optical_frame"
        point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        point_world = self.tf_buf.transform(point_s, "map")

        world_point = (point_world.point.x, point_world.point.y, point_world.point.z)
        
        if math.isnan(world_point[0]) or math.isnan(world_point[1] or math.isnan(world_point[2])):
            return
        
        t = self.tf_buf.lookup_transform("map", "base_link", rospy.Time(0))
        point_base = PoseStamped()
        point_base.pose.position.x = 0
        point_base.pose.position.y = 0
        point_base.pose.position.z = 0
        point_base.pose.orientation = t.transform.rotation
        point_base.header.frame_id = "base_link"
        point_base.header.stamp = rospy.Time(0)
        point_base = self.tf_buf.transform(point_base, "map", rospy.Duration(1))
        
        # Extract current yaw angle of the robot from quaternion
        quaternion = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
        euler = euler_from_quaternion(quaternion)
        self.current_yaw = euler[2]
        
        # Get coordinate of base in map frame
        rospy.loginfo("this is the point in the map frame: " + str(point_world))
        
        self.move(world_point, point_base)

        

    def move(self, point_world, point_base):
        
        #calculate euclidean distance from point_world to point_base
        dist = np.sqrt((point_world[0] - point_base.pose.position.x) ** 2 + (point_world[1] - point_base.pose.position.y) ** 2 + (point_world[2] - point_base.pose.position.z) ** 2)
        rospy.loginfo("dist: " + str(dist))
        
        #calculate angle to target
        angle_to_target = np.arctan2(point_world[1] - point_base.pose.position.y, point_world[0] - point_base.pose.position.x)
        rospy.loginfo("angle_to_target: " + str(angle_to_target))
        
        #calculate angle to base
        angle_to_base = np.arctan2(point_base.pose.position.y - point_world[1], point_base.pose.position.x - point_world[0])
        rospy.loginfo("angle_to_base: " + str(angle_to_base))
        
        #calculate angle difference
        #angle_diff = angle_to_target - angle_to_base
        angle_diff = angle_to_target - self.current_yaw
        rospy.loginfo("angle_diff: " + str(angle_diff))
        
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        angle_diff_degrees = angle_diff * 180 / np.pi

        Kp = 1.5
        angular_velocity = Kp * angle_diff

        #publish Twist message to rotate point_base to point_world
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angular_velocity

        self.cmd_vel_pub.publish(twist)
        rospy.sleep(2)
        
        
        twist = Twist()
        twist.linear.x = dist/2
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0

        if(dist < 0.3 or self.moves_completed != 0):
            twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(1)

def main():

    am = cylinder_inspector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

