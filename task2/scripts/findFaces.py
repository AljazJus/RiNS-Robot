

import sys
import rospy
import cv2
import numpy as np


from os.path import dirname, join

#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError

import rospy 
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
import actionlib
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import Point, Vector3, Quaternion
from std_msgs.msg import String, Bool, ColorRGBA

from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

from nav_msgs.srv import GetPlan, GetPlanRequest

from geometry_msgs.msg import PoseStamped

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler,euler_from_quaternion



class face_handle:
    def __init__(self, main):

        #array of detected faces
        self.faces=[]

        #radius around the face that is considered to be the face
        self.rad=0.4

        #
        self.main_prog=main

        #subscribing to the face markers
        
        
        #map data
        self.cv_map = None
        self.map_resolution = None
        self.map_transform = None
        self.bridge = CvBridge()

        #subscribing to the map data
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)


        #print out the list of faces detected
        rospy.loginfo("////////////////////FACES////////////////////////")
        for face in self.faces:
            rospy.loginfo("Found:{} ({}, {}) seen {}".format(face[0], face[1][0], face[1][1],face[2]))
        rospy.loginfo("////////////////////FACES////////////////////////")

        
    
    def face_detected(self,msg):
        """
        This function is called when a face is detected
        And it adds the face to the list of faces
        """

        #get the position of the face and distance from the robot
        #self.face_distance=mark.pose.orientation.w
        

        #get last face detected
        mark=msg.markers[-1]
        face_x,face_y=float(mark.pose.position.x),float(mark.pose.position.y)

        #check if the face detected is not nan
        if(math.isnan(face_x) or math.isnan(face_y)):
            return
        
        pred_faco=self.face_orientation(mark)

        #check if the face is already is empty
        if(self.faces==[]):
            rospy.loginfo("Add first face")
            #!move to the face zdej returnam 
            #self.move_to_face(pred_faco)
            #adds the face to the list of faces
            self.faces.append(["Face"+str(len(self.faces)),[face_x,face_y],1,pred_faco])
            return self.faces
        else:
            #check if the face is already in the list and has been added before
            for face in self.faces:
                #check if the face is close to the face in the list
                if(abs(face_x-face[1][0])<self.rad and abs(face_y-face[1][1])<self.rad):
                    if( self.simila_angle(face[3],pred_faco)):
                        rospy.loginfo("Edit a face"+face[0])
                        face[2]=face[2]+1
                        #this ajusts the position of the face in the list
                        #it does it 5 times to make sure it is in the right place
                        #first 5 times it seas the face 
                        if face[2]<5:
                            face[1][0]=(face[1][0]*((face[2]-1)/face[2])+face_x*(1/face[2]))
                            face[1][1]=(face[1][1]*((face[2]-1)/face[2])+face_y*(1/face[2]))
                        return self.faces
            #if it has not been added before add it to the list
            else:
                #!move to the face
                #self.move_to_face(pred_faco)
                rospy.loginfo("Add another face:"+str(len(self.faces)))
                self.faces.append(["Face"+str(len(self.faces)),[face_x,face_y],1,pred_faco])
                return self.faces


    def face_orientation(self,mark):
        """
        This function is called when the orientation of the face is detected
        """
        face_x,face_y=float(mark.pose.position.x),float(mark.pose.position.y)
        
        #get the current position of the robot
        curPos=self.get_current_pos()

        #calculate the point in aorund of the face
        tamp=[(face_x-0.5,face_y),(face_x+0.5,face_y),(face_x,face_y-0.5),(face_x,face_y+0.5)]
        tmp=[]
        #check if the point is reachable to the robot
        for p in tamp:
            if self.check_point_rad(p[0],p[1]):
                tmp.append(p)

        rospy.loginfo("FACE ORIENTATION")
        rospy.loginfo(tmp)
        #calculate the distance from the robot to the point  
        for i in range(len(tmp)):
            distance=math.sqrt((tmp[i][0] - curPos[0])**2 + (tmp[i][1] - curPos[1])**2)
            a=tmp[i]
            tmp[i]=[distance,a]

        #sort the points by distance
        tmp=sorted(tmp)
        
        if tmp==[]:
            #if there is no point to go to
            rospy.loginfo("NO POINTS TO GO TO!!!!!!!!!!!")
            return
        #choses the closest viable point
        point=tmp[0][1]
        
        #calculate orientation of the robot to face the point
        z, w = self.calculate_z_w( (face_x,face_y), point )
        point=point+(z,w)

        return point
        
    def get_current_pos(self):
        """this function gets the current position of the robot"""
        try:
            odom_msg = rospy.wait_for_message("/odom", Odometry)
            return(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
        except Exception as e:
            print(e)
            return 0
     
    def map_callback(self, data):
        """callback function for the map data"""
        try:
            self.cv_map = np.array(data.data).reshape((data.info.height, data.info.width))
            self.map_resolution = data.info.resolution
            self.map_transform = data.info.origin
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        #self.cv_map = np.flipud(self.cv_map)
        self.map_resolution = data.info.resolution
        self.map_transform = data.info.origin

    def check_point_rad(self, x, y):
        """
        Check if a point is reachable to the robot 
        If there are obstacles near the point it will return false
        """
        
        if (math.isnan(x) or math.isnan(y)):
            return False

        radius = 0.2 # radius to search for valid points
        res = self.map_resolution # map resolution

        # convert world coordinates to map coordinates
        x_map = int((x - self.map_transform.position.x) / res)
        y_map = int((y - self.map_transform.position.y) / res)
        
        # check if point is out of bounds
        if x_map < 0 or x_map >= self.cv_map.shape[1] or y_map < 0 or y_map >= self.cv_map.shape[0]:
            return False
        
        if self.cv_map[y_map, x_map]==-1:
            return False

        # find closest point with a value of 100 within radius
        for i in range(-int(radius / res), int(radius / res) + 1):
            for j in range(-int(radius / res), int(radius / res) + 1):
                if i ** 2 + j ** 2 > (radius / res) ** 2:
                    continue # skip points outside of circle
                x_curr = x_map + i
                y_curr = y_map + j
                if x_curr >= 0 and x_curr < self.cv_map.shape[1] and y_curr >= 0 and y_curr < self.cv_map.shape[0]:
                    data_val = self.cv_map[y_curr, x_curr]
                    if data_val == 100:
                        return False

        return True

    def calculate_z_w(self,point, bot_position):
        # point is a tuple containing (x, y) coordinates of the point
        # bot_position is a tuple containing (x, y) coordinates of the turtle bot

        dx = point[0] - bot_position[0]
        dy = point[1] - bot_position[1]

        angle = math.atan2(dy, dx)

        q = quaternion_from_euler(0, 0, angle)

        z = q[2]
        w = q[3]
        if w==0 and z==0:
            w=1
        return z, w
    
    def simila_angle(self, face,point):
        """checks if the face is in the same direction as the bot"""
        #print(face_orientation)
        #print(face)
        face_angle = euler_from_quaternion((0,0,face[2],face[3]))
        point_angle = euler_from_quaternion( (0,0,point[2],point[3]))


        if abs(face_angle[2] - point_angle[2]) < 0.7 :
            return True
        else:
            return False
        

    def approche_position(self,mark):
        """
        This function calculates the position infront of the object
        """

        #todo: calculate the position infront of the object
        ob_x,ob_y=mark

        #closest point from wall to mark
        
        wall_coord = self.find_closest_wall(mark)
        
        #calculate line from wall_cord to ob_x,ob_y
        center = np.array([ob_x, ob_y])
        wall = np.array([wall_coord[0], wall_coord[1]])


        # Calculate the direction vector from center to wall
        #!check if the object is ouut of the map
        print("----------------check point rad"+str(self.chech_if_point_reacheble((ob_x,ob_y))))
        if self.chech_if_point_reacheble((ob_x,ob_y))==0:
            direction = center - wall
        else:
            direction =  (center - wall )*-1
            

        # Calculate the point that is 0.5 meters away from the wall on the opposite side of the center
        opposite_point = wall + direction/np.linalg.norm(direction)*0.5
        
        #convert np.array to tuple
        opposite_point = tuple(opposite_point)

        #calculate orientation of the robot to face the point
        z, w = self.calculate_z_w( (ob_x,ob_y), opposite_point )
        point=opposite_point+(z,w)
        
        return point
    
    def chech_if_point_reacheble(self,mark):
        """checks if the point is reachable"""

        res = self.map_resolution # map resolution
        
        # convert world coordinates to map coordinates
        x_map = int((mark[0] - self.map_transform.position.x) / res)
        y_map = int((mark[1] - self.map_transform.position.y) / res)

        # check if point is out of bounds
        data_val = self.cv_map[y_map, x_map]
        if data_val == 100:
            return 1
        elif data_val == -1:
            return 2
        return 0          


    def find_closest_wall(self, mark, max_radius=0.5):
        res = self.map_resolution # map resolution
        
        # convert world coordinates to map coordinates
        x_map = int((mark[0] - self.map_transform.position.x) / res)
        y_map = int((mark[1] - self.map_transform.position.y) / res)
        
        # check if point is out of bounds
        #if x_map < 0 or x_map >= self.cv_map.shape[1] or y_map < 0 or y_map >= self.cv_map.shape[0]:
        #    return False

        closest_dist = float('inf')
        closest_point = None
        for radius in range(1, int(max_radius / res) + 1):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i ** 2 + j ** 2 > radius ** 2:
                        continue # skip points outside of circle
                    x_curr = x_map + i
                    y_curr = y_map + j
                    if x_curr >= 0 and x_curr < self.cv_map.shape[1] and y_curr >= 0 and y_curr < self.cv_map.shape[0]:
                        data_val = self.cv_map[y_curr, x_curr]
                        if data_val == 100:
                            # convert map coordinates to world coordinates
                            x_curr = x_curr * res + self.map_transform.position.x
                            y_curr = y_curr * res + self.map_transform.position.y

                            # calculate distance from current point to original point
                            dist = math.sqrt((x_curr - mark[0]) ** 2 + (y_curr - mark[1]) ** 2)

                            # update closest point if distance is smaller
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_point = (x_curr, y_curr)

        return closest_point

        