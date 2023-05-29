
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
        self.num_faces = 0
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
            if confidence > 140:
                print("Unknown face")
                return FaceRecognitionResponse(-1)
            print("Face identified as", label)
            return FaceRecognitionResponse(label)
        else:
            # Memorize a face
            faces = self.detect_faces()
            if len(faces) == 0:
                return FaceRecognitionResponse(-3)
            
            for face in faces:
                self.faces.append(face)
                self.labels.append(self.num_faces)
            self.num_faces += 1
            self.recognizer.train(self.faces, np.array(self.labels))
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

        transformations = [
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.flip(img, 0),
            cv2.flip(img, 1),
            cv2.flip(img, -1),
            cv2.GaussianBlur(img, (5, 5), 0),
            cv2.GaussianBlur(img, (9, 9), 0),
            cv2.GaussianBlur(img, (13, 13), 0),
            cv2.resize(img, (0, 0), fx=0.5, fy=0.5),
            cv2.resize(img, (0, 0), fx=1.5, fy=1.5),
            cv2.equalizeHist(img),
            cv2.bilateralFilter(img, 9, 75, 75),
            cv2.bilateralFilter(img, 13, 100, 100),
            cv2.bilateralFilter(img, 17, 125, 125),
        ]

        faces = []
        for transformation in transformations:
            face_cascade = cv2.CascadeClassifier(self.script_dir + '/haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("Error: Failed to load face detection classifier")
                raise Exception("Failed to load face detection classifier")

            faces_rect = face_cascade.detectMultiScale(transformation, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, w, h) in faces_rect:
                faces.append(transformation[y-10:y+h+10, x-10:x+w+10])
        print(len(faces))
        return faces
    
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
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        face_cascade = cv2.CascadeClassifier(self.script_dir + '/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Failed to load face detection classifier")
            raise Exception("Failed to load face detection classifier")

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [img[y-10:y+h+10, x-10:x+w+10] for (x, y, w, h) in faces]


if __name__ == '__main__':
    rospy.init_node('face_recognizer')
    face_recognizer = FaceRecognizer()
    rospy.spin()
