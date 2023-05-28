import rospy
import numpy as np
import matplotlib.pyplot as plt
import imageio
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion
from math import atan2
from task3.msg import Points_visited

class AutonomousNavigationGUI:
    def __init__(self):
        self.resolution = 0.050000
        self.origin = (-12.200000, -12.200000)
        self.image_size = (480, 480)
        self.max_dist_pixels_squared = 30**2   #dist = 30
        self.standing_points_radius = 7
        plt.set_cmap("gray")
        img = imageio.imread('/home/anej/FRI/Ris/ROS_ws_1/src/task3/maps/good_map1.pgm')
        self.original = np.array(img, dtype=np.uint8)
        self.img_matrix = np.array(img, dtype=np.uint8)
        self.img_matrix_colored = np.array([img, img, img], dtype=np.uint8)
        self.img_matrix_colored = np.transpose(self.img_matrix_colored, (1, 2, 0))
        self.current_pixel_colored = None
        self.current_pos_img = None
        self.b_x = (190, 340)    # bound x
        self.b_y = (170, 300)    # bound y
        self.points_all = self.get_points(self.img_matrix)

        for p in self.points_all:
            self.img_matrix[p[0], p[1]] = 1
            self.img_matrix_colored[p[0], p[1]] = [0, 0, 255]
        
        _, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.img_matrix_colored[self.b_y[0]:self.b_y[1], self.b_x[0]:self.b_x[1], :])
        
        self.pub_points_visited = rospy.Publisher('points_visited', Points_visited, queue_size=100)
        self.points_visited = [False] * len(self.points_all)
        self.points_visited_count = 0
        
        self.mainLoop()

    def mainLoop(self):
        while True:
            amcl_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
            pos = amcl_pose.pose.pose.position
            ori = amcl_pose.pose.pose.orientation
            z_orientation = euler_from_quaternion([0, 0, ori.z, ori.w])[2]

            (x_real, y_real) = (pos.x, pos.y)
            y_img, x_img = self.real_to_map_coordinates((y_real, x_real))

            for i in range(len(self.points_all)):
                if not self.points_visited[i]:
                    p = self.points_all[i]
                    dy = p[0] - y_img
                    dx = p[1] - x_img
                    dist = dy**2 + dx**2
                    if dist < self.max_dist_pixels_squared:
                        angle = atan2(-dy, dx)
                        diff = (angle - z_orientation)
                        if abs(diff) < 0.525:    # pusti 0.525!
                            line_pixels = self.get_line_pixels(x_img, y_img, p[1], p[0])
                            through_wall = False
                            for lp in line_pixels:
                                val = self.img_matrix[lp[1], lp[0]]
                                if val == 0 or val == 205:
                                    through_wall = True
                                    break
                            if through_wall:
                                continue
                            else:
                                self.img_matrix_colored[p[0], p[1]] = [0, 255, 0]
                                self.img_matrix[p[0], p[1]] = 2
                                self.points_visited_count += 1
                                self.points_visited[i] = True
            # za ohranjanje barv
            if self.current_pos_img is not None:
                self.img_matrix_colored[self.current_pos_img[0], self.current_pos_img[1], :] = self.current_pixel_colored
                
            self.current_pos_img = (y_img, x_img)
            self.current_pixel_colored = self.img_matrix_colored[y_img, x_img, :].copy()

            print(f"{self.points_visited_count} / {len(self.points_all)} = {round(self.points_visited_count / len(self.points_all) * 100, 2)}% of all points are visited")
            self.img_matrix_colored[y_img, x_img] = [255, 0, 0]
            self.im.set_data(self.img_matrix_colored[self.b_y[0]:self.b_y[1], self.b_x[0]:self.b_x[1], :])
            plt.pause(0.001)  # Pause for a short duration to update the figure
            plt.draw()

            self.pub_points_visited.publish(self.points_visited)

    def real_to_map_coordinates(self, point_real):
        #point_real = (y_real, x_real)
        y_map = self.image_size[0] - round((point_real[0] - self.origin[1]) / self.resolution)
        x_map = round((point_real[1] - self.origin[0]) / self.resolution)
        return (y_map, x_map)
    
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

if __name__ == '__main__':
    rospy.init_node('autonomous_navigation_GUI')
    autonomous_navigation = AutonomousNavigationGUI()
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