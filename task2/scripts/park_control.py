#!/usr/bin/python3

import rospy
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, ColorRGBA
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import sys
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Quaternion, Twist
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
import math
from nav_msgs.msg import Odometry

class park_controller():
    def __init__(self):

        rospy.init_node('park_controller', anonymous=True)
        
        # Set up the action client for the move_base action
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

        # Set up the publisher for the robot's velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
        
        self.result_of_park = None
        
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
        self.extend.points = [JointTrajectoryPoint(positions=[0,1.2,0.2,0.0],
                                                    time_from_start = rospy.Duration(1))]
        
        # new JointTrajectoryPoint(positions=[0,1.3,0.2,0.0],
        # old JointTrajectoryPoint(positions=[0,0.1,1,0.3]
        self.right = JointTrajectory()
        self.right.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.right.points = [JointTrajectoryPoint(positions=[-1.57,0.3,1,0],
                                                    time_from_start = rospy.Duration(1))]
        
        # rospy.init_node('image_converter', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)
        
        self.window_dim = 2

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1
        
        # self.arm_movement_pub.publish(self.extend)
        #rospy.sleep(5)

        self.search_for_parking = False
        rospy.sleep(1)
        self.arm_movement_pub.publish(self.extend)
        rospy.sleep(5)
        # Subscribe to the image and/or depth topic
        self.image_sub = rospy.Subscriber("/arm_camera/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/arm_camera/depth_registered/image_raw", Image, self.depth_callback)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.search_for_parking = True
        # call park function until we are done
    
    def get_pose(self,e,dist):
        rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        rospy.loginfo("ENTERED GET POSE FUNCTION")
        rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Calculate the position of the detected ellipse

        k_f = 525 # kinect focal length in pixels

        elipse_x = self.dims[1] / 2 - e[0][0]
        elipse_y = self.dims[0] / 2 - e[0][1]

        angle_to_target = np.arctan2(elipse_x,k_f)

        # Get the angles in the base_link relative coordinate system
        x,y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)

        ### Define a stamped message for transformation - directly in "base_frame"
        #point_b = PointStamped()
        #point_b.point.x = x
        #point_b.point.y = y
        #point_b.point.z = 0.3
        #point_b.header.frame_id = "base_link"
        #point_b.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        point_world = self.tf_buf.transform(point_s, "map")
        
        #point_base = self.tf_buf.transform(point_b, "map")
        
        # Get coordinate of base in map frame
        t = self.tf_buf.lookup_transform("map", "base_link", rospy.Time(0))
        point_base = PointStamped()
        point_base.point.x = 0
        point_base.point.y = 0
        point_base.point.z = 0
        point_base.header.frame_id = "base_link"
        point_base.header.stamp = rospy.Time(0)
        point_base = self.tf_buf.transform(point_base, "map", rospy.Duration(1.0))

        rospy.loginfo("this is the point in the map frame: " + str(point_world))
        #rospy.loginfo("The distance between base x and point x is: " + str(point_base.point.x - point_world.point.x))
        #rospy.loginfo("The distance between base y and point y is: " + str(point_base.point.y - point_world.point.y))

        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_world.point.x
        pose.position.y = point_world.point.y
        pose.position.z = point_world.point.z

        # Create a marker used for visualization
        self.marker_num += 1
        marker = Marker()
        marker.header.stamp = point_world.header.stamp
        marker.header.frame_id = point_world.header.frame_id
        marker.pose = pose
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Time(0)
        marker.id = self.marker_num
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0, 1, 0, 1)
        

        self.markers_pub.publish(marker)

        self.marker_array.markers.append(marker)
        # self.markers_pub.publish(self.marker_array)
        
        
        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_base.point.x
        pose.position.y = point_base.point.y
        pose.position.z = point_base.point.z

        # Create a marker used for visualization
        self.marker_num += 1
        marker = Marker()
        marker.header.stamp = point_base.header.stamp
        marker.header.frame_id = point_base.header.frame_id
        marker.pose = pose
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Time(0)
        marker.id = self.marker_num
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(1, 0, 0, 1)
        
        self.markers_pub.publish(marker)

        self.marker_array.markers.append(marker)
        rospy.loginfo("IM HERE")
        result = self.park(point_world, point_base)

        


    def image_callback(self,data):
        print('I got a new image!')

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = cv_image.shape

        # Tranform image to gayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        img = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example how to draw the contours, only for visualization purposes
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Contour window",img)
        cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)


        # Find two elipses with same centers
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                #             print dist
                if dist < 5:
                    candidates.append((e1,e2))

        print("Processing is done! found", len(candidates), "candidates for rings")

        try:
            depth_img = rospy.wait_for_message('/arm_camera/depth/image_raw', Image)
        except Exception as e:
            print(e)

        # Extract the depth from the depth image
        for c in candidates:
            rospy.loginfo("Processing candidate")
            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # rospy.loginfo("With center in: {} ---- {}".format(e1, e2,))
            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)
            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

            depth_image = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
            

            img_window = depth_image[x_min:x_max,y_min:y_max]
            
            #middle pixel index
            mpi = (round(img_window.shape[0] / 2), round(img_window.shape[1] / 2))
            middle_window = img_window[(mpi[0] - self.window_dim):(mpi[0] + self.window_dim), (mpi[1] - self.window_dim):(mpi[1] + self.window_dim)]
            rospy.loginfo(middle_window)
            
            #cv2.imwrite(f"ring_img_{len(self.rings) + 1}.jpg", img_window_color)
            
            valid_indexes = ~np.isnan(img_window)
            #img_window = img_window.reshape(-1)
            img_window = img_window[valid_indexes]
            
            self.result_of_park = self.get_pose(e1, float(np.mean(img_window)))

            if len(candidates)>0:
                cv2.imshow("Image window",cv_image)
                cv2.waitKey(1)

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
    
    
    def park(self, point_world, point_base):
        self.arm_movement_pub.publish(self.retract)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = point_world.point.x
        goal.target_pose.pose.position.y = point_world.point.y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.x = 0
        goal.target_pose.pose.orientation.y = 0
        goal.target_pose.pose.orientation.z = 0
        goal.target_pose.pose.orientation.w = 1
        
        self.move_base_client.send_goal(goal)
        rospy.loginfo("----------------------------------")
        rospy.loginfo("Sending goal to move_base: ")
        rospy.loginfo(goal)
        rospy.loginfo("----------------------------------")
        result = self.move_base_client.wait_for_result(rospy.Duration.from_sec(10.0))
        result = True
        rospy.loginfo(result)
        if not result:
            rospy.loginfo("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
            rospy.logerr("Did not reach goal")
            rospy.loginfo("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
            return False
        else:
            rospy.loginfo("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            rospy.loginfo("Reached goal")
            rospy.loginfo("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            self.search_for_parking = False
            self.image_sub.unregister()
            cv2.destroyAllWindows()
            return True

        

def main():

    am = park_controller()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

