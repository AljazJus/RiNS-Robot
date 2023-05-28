import rospy
import numpy as np
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
import actionlib
from task3.srv import GiveGoal
from task3.msg import Robot_pose

class Main:
    def __init__(self):
        self.state = 0
        self.goals = []
        self.move_client=actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move = True
        self.askRandom = True

        self.get_goal = rospy.Subscriber('move_robot', Robot_pose, self.move_robot_handle)

        while self.move:
            while len(self.goals) > 0:
                if self.move_to_goal(self.goals[0]):
                    rospy.loginfo("Reached the goal")
                else :
                    rospy.loginfo("Failed to reach the goal")
                del self.goals[0]
            if self.askRandom:
                print("I need a new goal")
                new_goal = self.get_new_goal_client()
                print("Got new goal from autonomous_navigation")
                if new_goal is not None:
                    self.goals.append(new_goal)
                else:
                    self.askRandom = False
                    print("There are no new points to explore")
            else:
                break

    def get_new_goal_client(self):
        rospy.wait_for_service('giveGoal')
        try:
            giveGoal = rospy.ServiceProxy('giveGoal', GiveGoal)
            resp = giveGoal()
            if resp.z == 0.0 and resp.w == 0.0:
                print("returning None")
                return None
            else:
                return [resp.x, resp.y, resp.z, resp.w]
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def move_robot_handle(self, req):
        print("I got a new goal")
        print(req)
        move_base_goal = [req.x, req.y, req.z, req.w]
        self.move_to_goal(move_base_goal)

    def move_to_goal(self ,point):
            
            x, y, z, w = point
            rospy.loginfo("Moving to ({}, {})".format(point[0], point[1]))

            client = self.move_client
            client.wait_for_server()
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.orientation.z = z
            goal.target_pose.pose.orientation.w = w

            client.send_goal(goal)
            if not client.wait_for_result(rospy.Duration.from_sec(40)):
                rospy.loginfo("The goal ({}, {}) cannot be reached".format(x, y))
                return False
            return True
if __name__ == '__main__':
    rospy.init_node('main_node')
    main = Main()
    rospy.spin()