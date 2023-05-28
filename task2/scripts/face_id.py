
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

        script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(script_path)
        print("The script is located at:", self.script_dir)

    def handle_face_recognition(self, req):
        if req.id:
            # Identify a face
            faces = self.detect_id()
            if len(faces) == 0:
                return FaceRecognitionResponse(-3)
            label, confidence = self.recognizer.predict(faces[0])
            print("Label:", label, "Confidence:", confidence)
            if confidence > 100:
                return FaceRecognitionResponse(-1)
            return FaceRecognitionResponse(label)
        else:
            # Memorize a face
            faces = self.detect_faces()
            if len(faces) == 0:
                return FaceRecognitionResponse(-3)
            self.faces.append(faces[0])
            self.labels.append(len(self.faces)-1)
            self.recognizer.update(self.faces, np.array(self.labels))
            print("Face memorized"+str(len(self.faces)-1))
            return FaceRecognitionResponse(-2)

    def detect_faces(self):

        try:
            imge = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0
        cv_img = None
        try:
            cv_img = self.bridge.imgmsg_to_cv2(imge, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(self.script_dir + '/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Failed to load face detection classifier")
            raise Exception("Failed to load face detection classifier")

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [img[y:y+h, x:x+w] for (x, y, w, h) in faces]
    
    def detect_id(self):

        try:
            imge = rospy.wait_for_message("/arm_camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0
        cv_img = None
        try:
            cv_img = self.bridge.imgmsg_to_cv2(imge, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(self.script_dir + '/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Failed to load face detection classifier")
            raise Exception("Failed to load face detection classifier")

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [img[y:y+h, x:x+w] for (x, y, w, h) in faces]

if __name__ == '__main__':
    rospy.init_node('face_recognizer')
    face_recognizer = FaceRecognizer()
    rospy.spin()
