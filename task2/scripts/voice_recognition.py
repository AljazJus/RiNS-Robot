#!/usr/bin/python3


import rospy
from std_msgs.msg import String, Bool
# import speech_recognition from ros-vosk package
from ros_vosk.msg import speech_recognition
import re

class voice_detector():
    def __init__(self):
        rospy.init_node('voice_detector', anonymous=True)

        self.voice_initializer = rospy.Subscriber("/voice_initializer", Bool, self.initialize_listen)
        ## publish a array of colors
        self.voice_pub = rospy.Publisher("/potential_cylinders", String, queue_size=1)
        self.colors = []
    
    def initialize_listen(self, msg):
        if msg.data:
            print("I have begun listening to your voice")
            try:
                message = rospy.wait_for_message("/speech_recognition/final_result", String, timeout=10)
                print("I have heard you say: ", message)
                colors = self.extract_possible_cylinders(message, self.colors)
                if len(colors) == 0:
                    print("No cylinders found")
                    # open text input terminal and wait for the user to write the colors of the cylinders with suspected thieves
                    message = input("Can you tell me where the thieves are?")
                    colors = self.extract_possible_cylinders(message, self.colors)
                    if len(colors) == 0:
                        self.voice_pub.publish("")
                        return
                    else:
                        self.voice_pub.publish(",".join(colors))
                        return
                else:
                    self.voice_pub.publish(",".join(colors))
                    return
            except rospy.exceptions.ROSException as e:
                print("Timeout occurred:", e)
                # open text input terminal and wait for the user to write the colors of the cylinders with suspected thieves
                message = input("Can you tell me where the thieves are?")

                colors = self.extract_possible_cylinders(message, self.colors)
                if len(colors) == 0:
                    self.voice_pub.publish("")
                    return
                else:
                    self.voice_pub.publish(",".join(colors))
                    return
    
    
    def extract_possible_cylinders(self, msg, colors = []):
        #check if msg is string or has msg.data attribute
        print("Extracting possible cylinders")

        if isinstance(msg, str):
            pattern = re.compile(r'\b(yellow|green|red|blue|Yellow|Green|Red|Blue)\b')
            matches = re.findall(pattern, msg)
            
            for match in matches:
                if match not in colors:
                    color = match.split()[0]
                    colors.append(color)
            
            if not colors:
                return []
            
        else:
            pattern = re.compile(r'\b(yellow|green|red|blue|Yellow|Green|Red|Blue)\b')
            matches = re.findall(pattern, msg.data)

            for match in matches:
                if match not in colors:
                    color = match.split()[0]
                    colors.append(color)

            if not colors:
                return []
        
        print("Colors: ", colors)
        return colors
    
def main():

    vd = voice_detector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main()

