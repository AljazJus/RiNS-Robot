#!/usr/bin/env python

import rospy
import sys
import cv2
from sensor_msgs.msg import Image
from task2.srv import ImageRecognition, ImageRecognitionResponse
from cv_bridge import CvBridge, CvBridgeError

class ImageRecognitionServer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_recognition_server')
        self.bridge = CvBridge()

        # Subscribe to the image topic
        self.image_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        # Wait for the first image to arrive
        self.latest_image = None
        while self.latest_image is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Create the image recognition service
        self.image_recognition_service = rospy.Service('image_recognition', ImageRecognition, self.image_recognition_callback)

    def image_callback(self, msg):
        # Store the latest image

        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError as e:
            print(e)
            return

        # Unsubscribe from the image topic
        self.image_subscriber.unregister()

    def image_recognition_callback(self, req):
        # Retrieve the latest image
    
        # Process the image data here
        # ...

        # Return the recognition results in the response
        response = ImageRecognitionResponse()
        response.wonted = False
        response.prize=0
        response.color = ""
        return response

    def run(self):
        # Spin the node to receive messages
        rospy.spin()

if __name__ == "__main__":
    image_recognition_server = ImageRecognitionServer()
    image_recognition_server.run()