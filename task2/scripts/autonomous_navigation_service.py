import rospy
import numpy as np
import matplotlib.pyplot as plt
import imageio
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler
from math import atan2, pi
from task3.srv import GiveGoal, GiveGoalResponse
from task3.msg import Points_visited
from nav_msgs.srv import GetPlan
import time

class AutonomousNavigationService:
    def __init__(self):
        self.resolution = 0.050000
        self.origin = (-12.200000, -12.200000)
        self.image_size = (480, 480)
        self.max_dist_pixels_squared_seen = 28**2   #dist = 28  naj bo malo manjse kot zgornji
        self.standing_points_radius = 7
        self.giveGoal_service = rospy.Service('giveGoal', GiveGoal, self.giveGoal_handle)
        img = imageio.imread('/home/anej/FRI/Ris/ROS_ws_1/src/task3/maps/good_map1.pgm')
        self.original = np.array(img, dtype=np.uint8)
        self.img_matrix = np.array(img, dtype=np.uint8)
        self.last_index_pose = -1
        self.last_index_rotation = -1
        self.b_x = (190, 340)    # bound x
        self.b_y = (170, 300)    # bound y

        self.points_all = self.get_points(self.img_matrix)
        self.points_standing = self.get_standing_points(self.img_matrix)
        self.points_seen_from_standing_points = self.get_points_seen(self.img_matrix, self.points_all, self.points_standing)
        print(f"all points len: {len(self.points_all)}")
        print(f"standing points len: {len(self.points_standing)}")
        print(f"seen from standing points len: {len(self.points_seen_from_standing_points)}")
        #print(self.points_seen_from_standing_points)

        for p in self.points_all:
            self.img_matrix[p[0], p[1]] = 1
        
    def real_to_map_coordinates(self, point_real):
        #point_real = (y_real, x_real)
        y_map = self.image_size[0] - round((point_real[0] - self.origin[1]) / self.resolution)
        x_map = round((point_real[1] - self.origin[0]) / self.resolution)
        return (y_map, x_map)
    
    def map_to_real_coordinates(self, point_map):
        #point_map = (y_map, x_map)
        y_real = (self.image_size[0] - point_map[0]) * self.resolution + self.origin[1]
        x_real = point_map[1] * self.resolution + self.origin[0]
        return (y_real, x_real)
    
    def get_line_pixels(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        line_pixels = []
        err = dx - dy

        while x0 != x1 or y0 != y1:
            line_pixels.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        line_pixels.append((x0, y0))  # Include the last pixel
        
        return line_pixels


    def get_points(self, img_matrix):
        jump = 5
        all_points = []
        for i in range(self.b_y[0], self.b_y[1], jump):
            for j in range(self.b_x[0], self.b_x[1], jump):
                valid = True
                for y_offset in range(-1, 2, 1):
                    for x_offset in range(-1, 2, 1):
                        if img_matrix[i + y_offset, j + x_offset] != 254:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    all_points.append((i, j))
        return all_points

    def get_standing_points(self, matrix):
        matrix_copy = matrix.copy()
        standing_points = []
        for i in range(self.b_y[0], self.b_y[1], 1):
            for j in range(self.b_x[0], self.b_x[1], 1):
                valid = True
                for y_offset in range(-self.standing_points_radius, self.standing_points_radius + 1, 1):
                    for x_offset in range(-self.standing_points_radius, self.standing_points_radius + 1, 1):
                        if matrix_copy[i + y_offset, j + x_offset] != 254:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    point = (i, j)
                    standing_points.append(point)
                    matrix_copy[i, j] = 0
        return standing_points

    def get_points_seen(self, matrix, all, standing):
        third_pi = pi / 3
        points_seen = []
        for sp in standing:
            points_current = [[],[],[],[],[],[]]
            for i, p in enumerate(all):
                dy = p[0] - sp[0]
                dx = p[1] - sp[1]
                dist_square = dy**2 + dx**2
                if dist_square < self.max_dist_pixels_squared_seen:
                    line_pixels = self.get_line_pixels(sp[1], sp[0], p[1], p[0])
                    through_wall = False
                    for lp in line_pixels:
                        val = matrix[lp[1], lp[0]]
                        if val != 254:
                            through_wall = True
                            break
                    if through_wall:
                        continue
                    else:
                        angle = atan2(-dy, dx) + pi
                        index = int(angle // third_pi)
                        index = max(0, index)
                        index = min(5, index)
                        points_current[index].append(i)
            points_seen.append(points_current)
        return points_seen
                        
    def giveGoal_handle(self, req):
        amcl_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
        points_visited = list(rospy.wait_for_message('/points_visited', Points_visited).points_visited)
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        start = PoseStamped()
        start.header = header
        start.pose = amcl_pose.pose.pose

        start_time = time.time()
        plan_distances = []

        for i, ps in enumerate(self.points_standing):
            y_real, x_real = self.map_to_real_coordinates(ps)

            goal_pose = Pose()
            goal_pose.position = Point(x_real, y_real, 0)
            goal_pose.orientation = Quaternion(0, 0, 0, 1)

            goal = PoseStamped()
            goal.header = header
            goal.pose = goal_pose

            rospy.wait_for_service('/move_base/make_plan')
            try:
                make_plan = rospy.ServiceProxy('move_base/make_plan', GetPlan)
                response = make_plan(start, goal, 0.02)
                poses = response.plan.poses
                total_dist = 0
                for i in range(1, len(poses)):
                    position_now = poses[i].pose.position
                    position_before = poses[i - 1].pose.position
                    total_dist += ((position_now.x - position_before.x)**2 + (position_now.y - position_before.y)**2)**0.5
                plan_distances.append(total_dist)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        diff = round(time.time() - start_time, 2)
        print("Ended calculating plan distances")
        print(f"Time: {diff}s")
        print(plan_distances)

        plan_distances = np.array(plan_distances)
        sorted_indices = np.argsort(plan_distances)

        current_index = 0
        path_limit = 2
        best_index_point = None
        best_index_rotation = None
        max_not_visited = 0

        while True:
            while current_index < len(plan_distances) and plan_distances[sorted_indices[current_index]] <= path_limit:
                current_seen = self.points_seen_from_standing_points[sorted_indices[current_index]]
                for i in range(6):
                    if self.last_index_pose != sorted_indices[current_index] or self.last_index_rotation != i:
                        arr_current = current_seen[i]
                        not_visited_counter = 0
                        for j in arr_current:
                            if not points_visited[j]:
                                not_visited_counter += 1
                        if not_visited_counter > max_not_visited:
                            max_not_visited = not_visited_counter
                            best_index_point = sorted_indices[current_index]
                            best_index_rotation = i
                current_index += 1

            self.last_index_pose = best_index_point
            self.last_index_rotation = best_index_rotation
            print(f"best index point: {best_index_point}")
            print(f"best index rotation: {best_index_rotation}")
            if best_index_point is None:
                if current_index < len(plan_distances):
                    path_limit += 2
                else:
                    print("There are no more points to explore")
                    return GiveGoalResponse(0.0, 0.0, 0.0, 0.0)
            else:
                point = self.points_standing[best_index_point]
                y_map, x_map = point
                y_real, x_real = self.map_to_real_coordinates((y_map, x_map))
                angle_z = (pi / 3) * best_index_rotation - (5 * pi / 6)
                _, _, z_robot, w_robot = quaternion_from_euler(0, 0, angle_z)
                return GiveGoalResponse(x_real, y_real, z_robot, w_robot)

if __name__ == '__main__':
    rospy.init_node('autonomous_navigation_service')
    autonomous_navigation = AutonomousNavigationService()
    rospy.spin()


"""
img_matrix
0 - stene
205 - nedosegljivo območje
254 - dosegljivo območje
1 - neobiskane točke
2 - obiskane točke
"""

# kinect rgb camera image 57 stopinj horizontalnega FOV (field of view) ker je 1 radian