import cv2
import time
import mediapipe as mp

class HandTracker():
    def __init__(self, mode = False, max_hands = 2,
                 detection_confidence = .5, tracking_confidence = .5):
        self.Hand = mp.solutions.hands.Hands(mode, 
                                            max_hands, 
                                            detection_confidence,
                                            tracking_confidence)
        self.draw = mp.solutions.drawing_utils
        
    def findHands(self, image):
        RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.Hand.process(RGBimage)
        return result, RGBimage
    
    def findPosition(self, image, hand_number = 0, draw=False):
        self.result, RGBimage = findHands(image)
        if self.result.multi_hand_landmarks:
            

