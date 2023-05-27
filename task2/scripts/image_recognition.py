#!/usr/bin/env python

import rospy
import sys
import cv2
from sensor_msgs.msg import Image
from task2.srv import ImageRecognition, ImageRecognitionResponse
from cv_bridge import CvBridge, CvBridgeError
import pytesseract
import numpy as np

import pytesseract
from PIL import Image as PILImage

import cv2.aruco as aruco
import sys

import os



class ImageRecognitionServer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_recognition_server')
        self.bridge = CvBridge()

        script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(script_path)
        print("The script is located at:", self.script_dir)
        # Subscribe to the image topic

        # Wait for the first image to arrive

        # Create the image recognition service
        self.image_recognition_service = rospy.Service('image_recognition', ImageRecognition, self.image_recognition_callback)



    def image_recognition_callback(self, req):

        try:
            img = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        print("Image received")
        response = ImageRecognitionResponse()

        # Load the face detection classifier
        face_cascade = cv2.CascadeClassifier(self.script_dir+'/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Failed to load face detection classifier")
            raise Exception("Failed to load face detection classifier")

        # Convert the input image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no faces are detected, set the response fields and return
        if len(faces) == 0:
            response.wonted = False
            response.prize = 0
            response.color = ""
            return response

        # Find the face with the largest area
        best_face = None
        max_area = 0
        for face in faces:
            area = face[2] * face[3]
            if area > max_area:
                max_area = area
                best_face = face

        # Extract the region below the face from the input image
        (x, y, w, h) = best_face
        region_below_face = gray[y+h:y+h+100, x-50:x+w+50]

        scale_factor = 3
        region_below_face = cv2.resize(region_below_face, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        thresh_value = 100
        block_size = 11
        # Apply thresholding to isolate the number and word regions
        thresh = cv2.adaptiveThreshold(region_below_face, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=block_size, C=2)
        thresh[thresh < thresh_value] = 0

        # Recognize the text using Tesseract
        number_text = pytesseract.image_to_string(thresh)
        # print(number_text)
        #cv2.imshow('Poster', thresh)
        #cv2.waitKey(0)
        # Filter the recognized text to extract words and digits

        digits = ""
        color = ""
        
        for s in number_text.split():
            if s.isdigit():
                digits += s
            elif s.lower() in ["green", "red", "yellow", "blue","black"]:
                color = s.lower()
            else:
                print("Unknown word: " + s)
            
            
            # print("Unknown word: " + s)

        words = number_text.split()

        # Set the response fields
        if len(digits) == 0 and len(color) == 0:
            response.wonted = False
        else: 
            response.wonted = True

        if digits!="":
            response.prize = int(digits)
        else:
            response.prize = 0

        response.color = color

        return response



    def run(self):
        # Spin the node to receive messages
        rospy.spin()

if __name__ == "__main__":
    image_recognition_server = ImageRecognitionServer()
    image_recognition_server.run()