import rospy
import math
from nav_msgs.msg import Odometry


def position_nexto(mark):
     """
     This function is used to move the robot next to the object
     """
     c_x,c_y=float(mark.pose.position.x),float(mark.pose.position.y)
     
     #get the current position of the robot
     curPos=get_current_pos()
     #calculate the point in aorund of the face
     tamp=[(c_x-0.5,c_y),(c_x+0.5,c_y),(c_x,c_y-0.5),(c_x,c_y+0.5)]
     tmp=[]
     #check if the point is reachable to the robot
     for p in tamp:
         if check_point_rad(p[0],p[1]):
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
     return point



def check_point_rad( x, y):
    """
    Check if a point is reachable to the robot 
    If there are obstacles near the point it will return false
    """
    
    if (math.isnan(x) or math.isnan(y)):
        return False
    radius = 0.2 # radius to search for valid points
    res = map_resolution # map resolution
    # convert world coordinates to map coordinates
    x_map = int((x - map_transform.position.x) / res)
    y_map = int((y - map_transform.position.y) / res)
    
    # check if point is out of bounds
    if x_map < 0 or x_map >= cv_map.shape[1] or y_map < 0 or y_map >= self.cv_map.shape[0]:
        return False
    
    if cv_map[y_map, x_map]==-1:
        return False
    # find closest point with a value of 100 within radius
    for i in range(-int(radius / res), int(radius / res) + 1):
        for j in range(-int(radius / res), int(radius / res) + 1):
            if i ** 2 + j ** 2 > (radius / res) ** 2:
                continue # skip points outside of circle
            x_curr = x_map + i
            y_curr = y_map + j
            if x_curr >= 0 and x_curr < cv_map.shape[1] and y_curr >= 0 and y_curr < self.cv_map.shape[0]:
                data_val = v_map[y_curr, x_curr]
                if data_val == 100:
                    return False
    return True

def get_current_pos():
    """this function gets the current position of the robot"""
    try:
        odom_msg = rospy.wait_for_message("/odom", Odometry)
        return(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
    except Exception as e:
        print(e)
        return 0
   
