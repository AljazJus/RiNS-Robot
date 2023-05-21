import rospy
from std_msgs.msg import String
# import speech_recognition from ros-vosk package
from ros_vosk.msg import speech_recognition
import re

def recognize_voice():
    """
    This function recognizes the voice from the microphone using ros-vosk library.
    It returns the colors of the cylinders with suspected thieves.
    """
    try:
        print("I have begun listening to your voice")
        # if it times out, open a ROS text terminal and wait for the user to write the colors of the cylinders with suspected thieves
        message = rospy.wait_for_message("/speech_recognition/final_result", String, timeout=40)
        colors = extract_possible_cylinders(message)
        if len(colors) == 0:
            print("No cylinders found")
            try:
                # open text input terminal and wait for the user to write the colors of the cylinders with suspected thieves
                message = input("Can you tell me where the thieves are?")
                colors = extract_possible_cylinders(message)
                if len(colors) == 0:
                    return False
                else:
                    return colors
            except KeyboardInterrupt:
                print("User input interrupted")
                return False
        else:
            return colors
    except rospy.exceptions.ROSException as e:
        print("Timeout occurred:", e)
        try:
            # open text input terminal and wait for the user to write the colors of the cylinders with suspected thieves
            message = input("Can you tell me where the thieves are?")
            colors = extract_possible_cylinders(message)
            if len(colors) == 0:
                return False
            else:
                return colors
        except KeyboardInterrupt:
            print("User input interrupted")
            return False

    
    
def extract_possible_cylinders(msg, colors = []):
    #check if msg is string or has msg.data attribute
    if isinstance(msg, str):
        pattern = re.compile(r'\b(yellow|green|red|blue)\b')
        matches = re.findall(pattern, msg)
        
        for match in matches:
            if match not in colors:
                colors.append(match)
        
        if not colors:
            return False
        
        return colors
    else:
        print(msg.data)
        pattern = re.compile(r'\b(yellow|green|red|blue)\b')
        matches = re.findall(pattern, msg.data)
        
        for match in matches:
            color = match.split()[0]
            colors.append(color)

        if not colors:
            return False
    
    return colors

rospy.init_node("voice_recognition_node")
colors_list = recognize_voice()
print(colors_list)