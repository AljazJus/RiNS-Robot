

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




class FindFace:
    def __init__(self):

        #initiate the node
        rospy.init_node('Find_Faces')
        #array of detected faces
        self.faces=[]

        #radius around the face that is considered to be the face
        self.rad=0.4

        #subscribing to the face markers
        self.face_sub=rospy.Subscriber("face_markers", MarkerArray, self.face_detected)

        #publisher for the face markers
        self.face_pub = rospy.Publisher("Faces_found", MarkerArray, queue_size=5)
        
        #publisher for the goal markers
        self.move_clienr=actionlib.SimpleActionClient('move_base', MoveBaseAction)

        #publisher for the goal markers
        self.make_plan_service = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        #publisher for sound
        self.soundhandle = SoundClient()

        #map data
        self.cv_map = None
        self.map_resolution = None
        self.map_transform = None
        self.bridge = CvBridge()

        #subscribing to the map data
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        
        #this is an array taht hold all the markers we eill draw on rviz
        self.markerArray = MarkerArray()

        #id of markers
        self.i=0

        #arrat of locations infront of the faces  
        self.newGoals=[]

        
        #array of goals
        self.goals =[
           
            # x: -0.10278880596160889
            # y: -0.3156306743621826
            # z: -0.035814481090678175
            # w: 0.9993584556825471
            (-0.10278880596160889, -0.3156306743621826, -0.035814481090678175, 0.9993584556825471),

            # x: -0.08167016506195068
            # y: -0.9307079315185547
            # z: -0.771586336839827
            # w: 0.6361246142086446
            (-0.08167016506195068, -0.9307079315185547, -0.771586336839827, 0.6361246142086446),

            # x: -1.0353431701660156
            # y: -0.06289887428283691
            # z: 0.9944823530281476
            # w: 0.1049040014279669
            (-1.0353431701660156, -0.06289887428283691, 0.9944823530281476, 0.1049040014279669),
            # x: -1.4328947067260742
            # y: -0.08281493186950684
            # z: -0.8556434375893374
            # w: 0.517565752064703
            (-1.4328947067260742, -0.08281493186950684, -0.8556434375893374, 0.517565752064703),
            # x: -1.0149805545806885
            # y: 1.5947914123535156
            # z: 0.8971746531646282
            # w: 0.44167594650255637
            (-1.0149805545806885, 1.5947914123535156, 0.8971746531646282, 0.44167594650255637),
            # x: -1.0149805545806885
            # y: 1.5947914123535156
            # z: 0.8971746531646282
            # w: 0.44167594650255637
            (-1.0149805545806885, 1.5947914123535156, 0.8971746531646282, 0.44167594650255637),

            # x: -1.0149805545806885
            # y: 1.5947914123535156
            # z: 0.8971746531646282
            # w: 0.44167594650255637
            (-1.0149805545806885, 1.5947914123535156, 0.8971746531646282, 0.44167594650255637),

            # x: -1.3768168687820435
            # y: 1.9692890644073486
            # z: 0.43751997472804754
            # w: 0.899208691969761
            (-1.3768168687820435, 1.9692890644073486, 0.43751997472804754, 0.899208691969761),


            #    x: 0.15961682796478271
            #     y: 1.9827066659927368
            #     z: 0.6905718388806782
            #     w: 0.7232638075729759
            (0.15961682796478271, 1.9827066659927368, 0.6905718388806782, 0.7232638075729759),

            # x: -0.08920907974243164
            # y: 1.7677351236343384
            # z: 0.492132089979682
            # w: 0.8705205373868156
            (-0.08920907974243164, 1.7677351236343384, 0.492132089979682, 0.8705205373868156),

          
           # x: 0.542794942855835
           # y: 2.020752429962158
           # z: 0.9531778262148796
           # w: 0.3024103695515035
           (0.542794942855835, 2.020752429962158, 0.9531778262148796, 0.3024103695515035),

           # x: 1.3778777122497559
           # y: 1.950562596321106
           # z: -0.007862214953451577
           # w: 0.9999690923103702
           (1.3778777122497559, 1.950562596321106, -0.007862214953451577, 0.9999690923103702),

           # x: 1.1759477853775024
           # y: 1.1275452375411987
           # z: -0.9490859112438822
           # w: 0.3150173536146377
           (1.1759477853775024, 1.1275452375411987, -0.9490859112438822, 0.3150173536146377),

           # x: 0.9991668462753296
           # y: 1.124552607536316
           # z: -0.1563922882776459
           # w: 0.9876950198149638
           (0.9991668462753296, 1.124552607536316, -0.1563922882776459, 0.9876950198149638),

           # x: 2.389066696166992
           # y: 1.728233814239502
           # z: 0.009051963931573504
           # w: 0.9999590301352258
           (2.389066696166992, 1.728233814239502, 0.009051963931573504, 0.9999590301352258),

           # x: 2.552042007446289
           # y: 1.044256329536438
           # z: -0.19242611521582118
           # w: 0.9813114644102287
           (2.552042007446289, 1.044256329536438, -0.19242611521582118, 0.9813114644102287),

           # x: 2.5799717903137207
           # y: 0.9684247970581055
           # z: 0.9990790240782923
           # w: 0.042908083699542326
           (2.5799717903137207, 0.9684247970581055, 0.9990790240782923, 0.042908083699542326),

           # x: 2.316070079803467
           # y: -0.1127772331237793
           # z: 0.9993204986943334
           # w: 0.03685838967329986
           (2.316070079803467, -0.1127772331237793, 0.9993204986943334, 0.03685838967329986),

           # x: 1.2136905193328857
           # y: -0.048691511154174805
           # z: -0.7330022047926338
           # w: 0.6802262621871769
           (1.2136905193328857, -0.048691511154174805, -0.7330022047926338, 0.6802262621871769),

           # x: 3.272860288619995
           # y: -0.6422595977783203
           # z: -0.6831572555220102
           # w: 0.7302712949497843
           (3.272860288619995, -0.6422595977783203, -0.6831572555220102, 0.7302712949497843),

           # x: 3.1212475299835205
           # y: -1.2015891075134277
           # z: 0.5413560883226987
           # w: 0.8407934262563822
           (3.1212475299835205, -1.2015891075134277, 0.5413560883226987, 0.8407934262563822),

           # x: 1.8883428573608398
           # y: -1.605823040008545
           # z: -0.6733358179132538
           # w: 0.7393367813892999
           (1.8883428573608398, -1.605823040008545, -0.6733358179132538, 0.7393367813892999),

           # x: 2.0061328411102295
           # y: -1.6888556480407715
           # z: -0.39298204849993856
           # w: 0.9195461432450206
           (2.0061328411102295, -1.6888556480407715, -0.39298204849993856, 0.9195461432450206),

           # x: 1.192490816116333
           # y: -1.9960026741027832
           # z: 0.7276675434200587
           # w: 0.6859299864075173
           (1.192490816116333, -1.9960026741027832, 0.7276675434200587, 0.6859299864075173),

           # x: 0.027308344841003418
           # y: -1.0728521347045898
           # z: -0.8391159544804025
           # w: 0.5439525851914329
           (0.027308344841003418, -1.0728521347045898, -0.8391159544804025, 0.5439525851914329),
        ]

        #variable for the next goal
        self.nextGoal=0

        #status of the robot
        self.move_status="Waiting"

        #this goes through the list of goals and moves the robot to each one
        while len(self.goals)>self.nextGoal:

            if self.move_status=="Waiting":
                #self.greet_face()
                if self.move_to_goal(self.goals[self.nextGoal]):
                    rospy.loginfo("Reached the goal")
                else :
                    rospy.loginfo("Failed to reach the goal")
                self.nextGoal=self.nextGoal+1


        #print out the list of faces detected
        rospy.loginfo("////////////////////FACES////////////////////////")
        for face in self.faces:
            rospy.loginfo("Found:{} ({}, {}) seen {}".format(face[0], face[1][0], face[1][1],face[2]))
        rospy.loginfo("////////////////////FACES////////////////////////")
        
    
        #this draws all the markers 
        self.face_pub.publish(self.markerArray)
        
    
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
            self.move_to_face(pred_faco)
            #adds the face to the list of faces
            self.faces.append(["Face"+str(len(self.faces)),[face_x,face_y],1,pred_faco])
        else:
            #check if the face is already in the list and has been added before
            for face in self.faces:
                #check if the face is close to the face in the list
                if((face_x-face[1][0])<self.rad and (face_y-face[1][1])<self.rad):
                    if( self.simila_angle(face[3],pred_faco)):
                        rospy.loginfo("Edit a face"+face[0])
                        face[2]=face[2]+1
                        #this ajusts the position of the face in the list
                        #it does it 5 times to make sure it is in the right place
                        #first 5 times it seas the face 
                        if face[2]<5:
                            face[1][0]=(face[1][0]*((face[2]-1)/face[2])+face_x*(1/face[2]))
                            face[1][1]=(face[1][1]*((face[2]-1)/face[2])+face_y*(1/face[2]))
                        break
            #if it has not been added before add it to the list
            else:
                self.move_to_face(pred_faco)
                rospy.loginfo("Add another face:"+str(len(self.faces)))
                self.faces.append(["Face"+str(len(self.faces)),[face_x,face_y],1,pred_faco])
        

        #this draws all the markers
        #and clear then to draw new ones       
        self.markerArray = MarkerArray()
        self.markerArray.markers = []
        self.face_pub.publish(self.markerArray)
        self.markerArray = MarkerArray()
        #reset the id of faces
        #imporatant fot the markers to disapear
        self.i=0

        rospy.loginfo("FACEPRIT")
        #this puts all the marekers to mareker array 
        for face in self.faces:
            self.add_maeker(face[1],ColorRGBA(0, 1, 1, 1),face[0])
            self.add_arrow(face[3],ColorRGBA(0, 1, 0, 1))

        #darw the markers
        self.face_pub.publish(self.markerArray)
        

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
        
                
    def move_to_face(self,point):
        """
        This function moves the robot to the point that it calculates infron of the face
        """

        rospy.loginfo("NEW POINT FROM FACE RADIUS")
        rospy.loginfo(point)
        #self.newGoals.append(point)
        #appernd to the list of new goals
        self.goals.insert(self.nextGoal,point)
        #moves to the new goal 
        self.move_to_goal(point)

        #greets the face "HELO"
        self.greet_face()
        


    def get_current_pos(self):
        """this function gets the current position of the robot"""
        try:
            odom_msg = rospy.wait_for_message("/odom", Odometry)
            return(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
        except Exception as e:
            print(e)
            return 0
              
    def move_to_goal(self ,point):
            """this function moves the robot to the point
            It calculates the path and moves to the goal"""
        
            self.move_status="Moving"
            x,y,=point[0],point[1]
            #service movae base make plan 
            
            rospy.loginfo("Moving to ({}, {})".format(point[0], point[1]))

            client = self.move_clienr
            client.wait_for_server()
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.orientation.z = 0
            goal.target_pose.pose.orientation.w = 1

            if(len(point)> 2):
                z,w = point[2],point[3]
                goal.target_pose.pose.orientation.z = z
                goal.target_pose.pose.orientation.w = w
            

            client.send_goal(goal)
            if not client.wait_for_result(rospy.Duration.from_sec(40)):
                rospy.loginfo("The goal ({}, {}) cannot be reached".format(x, y))
                self.move_status="Waiting"
                return False
            self.move_status="Waiting"
            return True
           
    def add_maeker(self,point,color,name="Tese"):
        """
        Add a marker to the marker array
        """
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'map'
        marker.pose.position = Point(point[0], point[1], 0.0)
        marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Time(0)
        marker.id = self.i
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = color
        self.markerArray.markers.append(marker)
        self.i=self.i+1


        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'map'
        marker.pose.position = Point(point[0], point[1]+0.2, 2)
        marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Time(0)
        marker.id = self.i
        marker.scale = Vector3(0.3, 0.3, 0.3)
        marker.text = name
        marker.color = ColorRGBA(0, 0, 0, 1)
        self.markerArray.markers.append(marker)
        self.i=self.i+1

    def greet_face(self):
        """greets the face with a voice"""
        rospy.loginfo("Greeting face")
        
        voice = 'voice_kal_diphone'
        self.soundhandle.say('Hello', voice)

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
    
    def add_arrow(self,point,color):
        """adds an arrow to the point"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color = color
        marker.pose.orientation.w = point[3]
        marker.pose.orientation.z = point[2]
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        self.markerArray.markers.append(marker)
        marker.id = self.i
        self.i=self.i+1

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

if __name__ == '__main__':
  goal_mover = FindFace()

  rospy.spin()
        