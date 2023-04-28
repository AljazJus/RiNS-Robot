#!/usr/bin/python3

import sys
import rospy
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

class The_Ring:
    def __init__(self):
        rospy.init_node('image_converter', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Marker array object used for visualizations
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('rings', Marker, queue_size=100)
        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        
        self.min_limit = 0.2
        self.rings = []
        self.green_ring = None
        self.window_dim = 3
        self.epsilon = 0.01
        self.colors = ( (np.array([0.22023449, 0.95504346, 0.19155043]), 'green'),
                        (np.array([0.18645276, 0.18645276, 0.18645276]), 'black'),
                        (np.array([0.36964507, 0.6392554, 0.97609829]), 'blue'),
                        (np.array([0.47018454, 0.23760092, 0.22946943]), 'red'))
        #self.color_names = ('green', 'black', 'blue', 'red')
    
    def nearest_neighbour(self, col):
        min_dist = 100
        best_index = -1
        for i in range(len(self.colors)):
            dist = np.sum((self.colors[i][0] - col) ** 2)
            if dist < min_dist:
                best_index = i
                min_dist = dist
        return self.colors[best_index][1]
    
    def get_pose(self,e,dist, marker_color, color_name):
        print("dist:", dist)
        # Calculate the position of the detected ellipse

        k_f = 525 # kinect focal length in pixels

        elipse_x = self.dims[1] / 2 - e[0][0]
        elipse_y = self.dims[0] / 2 - e[0][1]

        angle_to_target_x = np.arctan2(elipse_x,k_f)
        angle_to_target_y = np.arctan2(-elipse_y,k_f)

        # Get the angles in the base_link relative coordinate system
        x,y,z = dist*np.cos(angle_to_target_x), dist*np.sin(angle_to_target_x), dist * np.sin(angle_to_target_y)

        ### Define a stamped message for transformation - directly in "base_frame"
        #point_s = PointStamped()
        #point_s.point.x = x
        #point_s.point.y = y
        #point_s.point.z = 0.3
        #point_s.header.frame_id = "base_link"
        #point_s.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = z
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        point_world = self.tf_buf.transform(point_s, "map")

        world_point = (point_world.point.x, point_world.point.y, point_world.point.z)
        if math.isnan(world_point[0]) or math.isnan(world_point[1] or math.isnan(world_point[2])):
            return
        
        for r in self.rings:
            if  abs(world_point[0] - r[0][0]) < self.min_limit and \
                abs(world_point[1] - r[0][1]) < self.min_limit and \
                abs(world_point[2] - r[0][2]) < self.min_limit:
                return
        
        self.rings.append((world_point, color_name))
        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = world_point[0]
        pose.position.y = world_point[1]
        pose.position.z = world_point[2]

        pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        print(self.rings)

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
        marker.color = marker_color
        

        self.markers_pub.publish(marker)

        rospy.loginfo("----------------------------------")
        rospy.loginfo("Current length of rings: %d", len(self.rings))
        rospy.loginfo("----------------------------------")
        rospy.loginfo("The current rings are: ")
        for r in self.rings:
            rospy.loginfo(r[1])
        rospy.sleep(5)
        
        if len(self.rings)>=1:
            rospy.loginfo("Fetch the green ring")
            if self.green_ring is None:
                for r in self.rings:
                    if r[1] == 'green':
                        self.green_ring = r
                        rospy.loginfo("----------------------------------")
                        rospy.loginfo("GOT THE GREEN RING !!!!!!!!!!!!!!!")
                        rospy.loginfo("----------------------------------")
                        rospy.loginfo(self.green_ring)
                        rospy.sleep(5)
                        break
                
        if self.green_ring is not None:
            rospy.loginfo("------------------------------------------------------")
            rospy.loginfo("Location of green is: %f %f %f", self.green_ring[0][0], self.green_ring[0][1] , self.green_ring[0][2])
            rospy.loginfo("------------------------------------------------------")
            
            curPos=self.get_current_pos()
            
            self.ring_publisher.publish(self.green_ring)
            
            rospy.sleep(5)


    def image_callback(self,data):

        odom = rospy.wait_for_message('/odom', Odometry)
        x = odom.twist.twist.linear.x
        y = odom.twist.twist.linear.y
        z = odom.twist.twist.angular.z
        if abs(x) > self.epsilon or abs(y) > self.epsilon or abs(z) > self.epsilon:
            return 
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = cv_image.shape

        # Tranform image to gayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        

        # Do histogram equlization
        img = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        """
        # Example how to draw the contours, only for visualization purposes
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Contour window",img)
        cv2.waitKey(1)
        """

        # Fit elipses to all extracted contours
        elps = []
        #print("Elipses:")
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                #rint(ellipse)
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

        #print("Processing is done! found", len(candidates), "candidates for rings")

        try:
            depth_img = rospy.wait_for_message('/camera/depth/image_raw', Image)
        except Exception as e:
            print(e)

        # Extract the depth from the depth image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            """
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)
            """

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
            #print(middle_window)
            
            not_nan_counter = np.sum(~np.isnan(np.array(middle_window)))
            if not_nan_counter != 0:
                continue
            
            img_window_color = cv_image[x_min:x_max,y_min:y_max]
            #cv2.imwrite(f"ring_img_{len(self.rings) + 1}.jpg", img_window_color)

            print("image_window_color_shape", img_window_color.shape)
            valid_indexes = ~np.isnan(img_window)
            #img_window = img_window.reshape(-1)
            img_window = img_window[valid_indexes]
            img_window_color_valid = img_window_color[valid_indexes]
            print("img_window color_valid", img_window_color_valid.shape)
            c = np.mean(img_window_color_valid, axis=0) / 255
            print(c)
            color_name = self.nearest_neighbour(c)
            self.get_pose(e1, float(np.mean(img_window)), ColorRGBA(c[0], c[1], c[2], 1), color_name)
        
        """
        if len(candidates)>0:
                cv2.imshow("Image window",cv_image)
                cv2.waitKey(1)
        """
        
    """
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
    """


def main():

    ring_finder = The_Ring()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
