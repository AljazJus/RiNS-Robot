
#!/usr/bin/env python

import rospy
import sys
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2.aruco as aruco
import os

from task2.srv import FaceRecognition, FaceRecognitionResponse

class FaceRecognizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.faces = []
        self.labels = []
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.service = rospy.Service('face_recognition', FaceRecognition, self.handle_face_recognition)

    def handle_face_recognition(self, req):
        if req.id:
            # Identify a face
            img = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)
            if len(faces) == 0:
                return FaceRecognitionResponse(-1)
            label, confidence = self.recognizer.predict(faces[0])
            return FaceRecognitionResponse(label)
        else:
            # Memorize a face
            img = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)
            if len(faces) == 0:
                return FaceRecognitionResponse(-1)
            self.faces.append(faces[0])
            self.labels.append(req.label)
            self.recognizer.update(self.faces, np.array(self.labels))
            return FaceRecognitionResponse(0)

    def detect_faces(self, img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [img[y:y+h, x:x+w] for (x, y, w, h) in faces]

if __name__ == '__main__':
    rospy.init_node('face_recognizer')
    face_recognizer = FaceRecognizer()
    rospy.spin()
