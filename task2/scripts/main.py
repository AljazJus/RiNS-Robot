

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

from findFaces import face_handle


class Main_task:

    def __init__(self):

        #initiate the node
        rospy.init_node('Main_task')

        print(sys.path)
        #face detection/ handeling object
        self.face=face_handle(self)

        #array of detected faces
        self.faces=[]
        self.face_count=0
        
        #array of detected cylinders
        self.cylinders=[]
        self.cylinders_markers=[]

        #array of detected rings
        self.rings=[]
        self.rings_markers=[]

        #radius around the face that is considered to be the face
        self.rad=0.4

        # subscribing to the cilinder detection
        self.cylinder_sub=rospy.Subscriber("cyliders", Marker, self.cylinder_handle)

        # subscribing to the ring detection
        self.ring_sub=rospy.Subscriber("rings", Marker, self.ring_handle)

        #subscribing to the face markers
        self.face_sub=rospy.Subscriber("face_markers", MarkerArray, self.face_handle)

        #publisher for the face markers
        self.face_pub = rospy.Publisher("Faces_found", MarkerArray, queue_size=5)
        
        #publisher for the goal markers
        self.move_clienr=actionlib.SimpleActionClient('move_base', MoveBaseAction)

        #publisher for the goal markers
        self.make_plan_service = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        #parking piblisher 
        self.park_pub = rospy.Publisher("park_initiated", Bool, queue_size=5)

        #publisher for sound
        self.soundhandle = SoundClient()

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

            (-1.3768168687820435, 1.9692890644073486, 0.43751997472804754, 0.899208691969761),

            (0.15961682796478271, 1.9827066659927368, 0.6905718388806782, 0.7232638075729759),

            (-0.08920907974243164, 1.7677351236343384, 0.492132089979682, 0.8705205373868156),

           (0.542794942855835, 2.020752429962158, 0.9531778262148796, 0.3024103695515035),

        #   (1.3778777122497559, 1.950562596321106, -0.007862214953451577, 0.9999690923103702),


        #    (1.1759477853775024, 1.1275452375411987, -0.9490859112438822, 0.3150173536146377),

        #    (0.9991668462753296, 1.124552607536316, -0.1563922882776459, 0.9876950198149638),

        #    (2.389066696166992, 1.728233814239502, 0.009051963931573504, 0.9999590301352258),


        #    (2.552042007446289, 1.044256329536438, -0.19242611521582118, 0.9813114644102287),

        #    (2.5799717903137207, 0.9684247970581055, 0.9990790240782923, 0.042908083699542326),

        #    (2.316070079803467, -0.1127772331237793, 0.9993204986943334, 0.03685838967329986),

        #    (1.2136905193328857, -0.048691511154174805, -0.7330022047926338, 0.6802262621871769),

        #    (3.272860288619995, -0.6422595977783203, -0.6831572555220102, 0.7302712949497843),

        #    (3.1212475299835205, -1.2015891075134277, 0.5413560883226987, 0.8407934262563822),

        #    (1.8883428573608398, -1.605823040008545, -0.6733358179132538, 0.7393367813892999),

        #    (2.0061328411102295, -1.6888556480407715, -0.39298204849993856, 0.9195461432450206),
        #    (1.192490816116333, -1.9960026741027832, 0.7276675434200587, 0.6859299864075173),
        #    (0.027308344841003418, -1.0728521347045898, -0.8391159544804025, 0.5439525851914329),
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

        rospy.loginfo("////////////////////RINGS////////////////////////")
        for i,ring in enumerate(self.rings):
            rospy.loginfo("Found:{} ({}, {}) seen color {}".format(i, ring[0][0], ring[0][1],ring[1]))
        rospy.loginfo("////////////////////RINGS////////////////////////")

        rospy.loginfo("////////////////////cylinders////////////////////////")
        for i,cylinder in enumerate(self.cylinders):
            rospy.loginfo("Found:{} ({}, {}) seen ".format(i, cylinder[0][0], cylinder[0][1]))
        rospy.loginfo("////////////////////cylinders////////////////////////")
        
        self.face_sub.unregister()

        for i,ring in enumerate(self.rings):
            if ring[1] == "green":
                pos=self.face.approche_position(ring[0])
                if self.move_to_goal(pos):
                    rospy.loginfo("Reached the goal")
                    self.add_maeker(pos,ColorRGBA(0, 1, 1, 1))
                    #new_msg=Bool.data(True)
                    self.park_pub.publish(True)
                break
        else :
            print("No green ring found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #this draws all the markers 
        self.face_pub.publish(self.markerArray)
        rospy.sleep(10)
    

    def ring_handle(self,msg):
        #TODO: read the correct color 

        color=msg.ns.split(":")[0]
        x=msg.pose.position.x
        y=msg.pose.position.y
        if(x is None or y is None):
            return
        
        self.rings.append([(x,y),color])
        self.rings_markers.append(msg)
        self.marker_update()

    def cylinder_handle(self,msg):
        
        #rospy.loginfo(msg)
        x=msg.pose.position.x
        y=msg.pose.position.y

        if(math.isnan(y) or math.isnan(x)):
            rospy.loginfo(msg)
            return
        #check if the face is already in the list and has been added before
        for i,cyl in enumerate(self.cylinders):
            cx=cyl[0][0]
            cy=cyl[0][1]
            
            #check if the new c  is close to the cyl in the list
            if(abs(x-cx)<0.5 and abs(y-cy)<0.5):
                rospy.loginfo("Edit a cylinder "+str(i)+"("+str(cx)+" , "+str(cy)+")")
                #ajust the position of the cylinder
                self.cylinders_markers[i].pose.position.x=(x*0.5+cx*0.5)
                self.cylinders_markers[i].pose.position.y=(y*0.5+cy*0.5)
                break
        else:
            color=msg.color
            self.cylinders.append([(x,y),color])
            self.cylinders_markers.append(msg)

        self.marker_update()
    
    def face_handle(self,msg):
        new_faces=self.face.face_detected(msg)

        if new_faces is None:
            return
        
        if(len(new_faces)>self.face_count):
            self.face_count=self.face_count+1
            self.move_to_face(new_faces[-1][3])
        
        self.faces=new_faces 

        self.marker_update()

    def marker_update(self):
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
        #this puts all the face marekers to mareker array 
        for face in self.faces:
            self.add_maeker(face[1],ColorRGBA(0, 1, 1, 1),face[0])
            self.add_arrow(face[3],ColorRGBA(0, 1, 0, 1))

        #this puts all the cylinder markers to marker array
        for cylinder in self.cylinders_markers:
            self.markerArray.markers.append(cylinder)
        
        #this puts all the ring markers to marker array
        for ring in self.rings_markers:
            self.markerArray.markers.append(ring)

        #darw the markers SPHERE
        self.face_pub.publish(self.markerArray)
    
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


if __name__ == '__main__':
  goal_mover = Main_task()

  rospy.spin()
        