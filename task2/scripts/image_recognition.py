#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from import ImageRecognition, ImageRecognitionResponse

class ImageRecognitionServer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_recognition_server')

        # Subscribe to the image topic
        self.image_subscriber = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # Wait for the first image to arrive
        self.latest_image = None
        while self.latest_image is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Create the image recognition service
        self.image_recognition_service = rospy.Service('image_recognition', ImageRecognition, self.image_recognition_callback)

    def image_callback(self, msg):
        # Store the latest image
        self.latest_image = msg

        # Unsubscribe from the image topic
        self.image_subscriber.unregister()

    def image_recognition_callback(self, req):
        # Retrieve the latest image
        image_data = self.latest_image.data

        # Process the image data here
        # ...

        # Return the recognition results in the response
        response = ImageRecognitionResponse()
        response.recognition_results = "Object detected"
        return response

    def run(self):
        # Spin the node to receive messages
        rospy.spin()

if __name__ == "__main__":
    image_recognition_server = ImageRecognitionServer()
    image_recognition_server.run()