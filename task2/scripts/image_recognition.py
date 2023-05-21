#!/usr/bin/env python

import rospy
import sys
import cv2
from sensor_msgs.msg import Image
from task2.srv import ImageRecognition, ImageRecognitionResponse
from cv_bridge import CvBridge, CvBridgeError
import pytesseract
import numpy as np

class ImageRecognitionServer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_recognition_server')
        self.bridge = CvBridge()

        # Subscribe to the image topic

        # Wait for the first image to arrive
        


        self.dictm = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        self.params =  cv2.aruco.DetectorParameters_create()

        # Create the image recognition service
        self.image_recognition_service = rospy.Service('image_recognition', ImageRecognition, self.image_recognition_callback)



    def image_recognition_callback(self, req):

        try:
            img = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0
        
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        response = ImageRecognitionResponse()
        response.wonted = False
        response.prize=0
        response.color = ""
        # Retrieve the latest image

        # Process the image data here
        # ...
        corners, ids, rejected_corners = cv2.aruco.detectMarkers(rgb_image,self.dictm,parameters=self.params)
        
        # Increase proportionally if you want a larger image
        image_size=(351,248,3)
        marker_side=50

        img_out = np.zeros(image_size, np.uint8)
        out_pts = np.array([[marker_side/2,img_out.shape[0]-marker_side/2],
                        [img_out.shape[1]-marker_side/2,img_out.shape[0]-marker_side/2],
                        [marker_side/2,marker_side/2],
                        [img_out.shape[1]-marker_side/2,marker_side/2]])

        src_points = np.zeros((4,2))
        cens_mars = np.zeros((4,2))

        if not ids is None:
            if len(ids)==4:
                print('4 Markers detected')

                cen_point = []
                for idx in ids:
                    # Calculate the center point of all markers
                    cors = np.squeeze(corners[idx[0]-1])
                    cen_mar = np.mean(cors,axis=0)
                    cens_mars[idx[0]-1]=cen_mar
                    cen_point = np.mean(cens_mars,axis=0)
            
                for coords in cens_mars:
                    #  Map the correct source points
                    if coords[0]<cen_point[0] and coords[1]<cen_point[1]:
                        src_points[2]=coords
                    elif coords[0]<cen_point[0] and coords[1]>cen_point[1]:
                        src_points[0]=coords
                    elif coords[0]>cen_point[0] and coords[1]<cen_point[1]:
                        src_points[3]=coords
                    else:
                        src_points[1]=coords

                h, status = cv2.findHomography(src_points, out_pts)
                img_out = cv2.warpPerspective(cv_image, h, (img_out.shape[1],img_out.shape[0]))
                
                ################################################
                #### Extraction of digits starts here
                ################################################
                
                # Cut out everything but the numbers
                img_out = img_out[125:221,50:195,:]
                
                # Convert the image to grayscale
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
                
                # Option 1 - use ordinairy threshold the image to get a black and white image
                #ret,img_out = cv2.threshold(img_out,100,255,0)

                # Option 1 - use adaptive thresholding
                img_out = cv2.adaptiveThreshold(img_out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
                
                # Use Otsu's thresholding
                #ret,img_out = cv2.threshold(img_out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
                # Pass some options to tesseract
                config = '--psm 13 outputbase nobatch digits'
                
                # Visualize the image we are passing to Tesseract
                cv2.imshow('Warped image',img_out)
                cv2.waitKey(1)
            
                # Extract text from image
                text = pytesseract.image_to_string(img_out, config = config)
                
                # Check and extract data from text
                print('Extracted>>',text)
                
                # Remove any whitespaces from the left and right
                text = text.strip()
                
        #             # If the extracted text is of the right length
                if len(text)==2:
                    x=int(text[0])
                    y=int(text[1])
                    print('The extracted datapoints are x=%d, y=%d' % (x,y))
                else:
                    print('The extracted text has is of length %d. Aborting processing' % len(text))
                
            else:
                print('The number of markers is not ok:',len(ids))
        else:
            print('No markers found')
        # Return the recognition results in the response
        
        return response

    def run(self):
        # Spin the node to receive messages
        rospy.spin()

if __name__ == "__main__":
    image_recognition_server = ImageRecognitionServer()
    image_recognition_server.run()