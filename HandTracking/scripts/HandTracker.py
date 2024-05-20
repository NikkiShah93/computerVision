import cv2
import time
import mediapipe as mp

class HandTracker():
    """
    This class will find the hands in any given image.
    It will return the position of the found points for each ID.
    """
    def __init__(self, mode = False, max_hands = 2, complexity = 1,
                 detection_confidence = 0.5, tracking_confidence = 0.5):
        self.hands_solutions = mp.solutions.hands
        self.Hand = self.hands_solutions.Hands(mode, max_hands, 
                                               complexity,
                                               detection_confidence, 
                                               tracking_confidence)
        self.draw = mp.solutions.drawing_utils
        
    def findHands(self, image):
        RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.Hand.process(RGBimage)
        return result, RGBimage
    
    def findPosition(self, image, hand_number = 0, draw=False):
        self.result, RGBimage = self.findHands(image)
        position_dict = {}
        h, w, c = image.shape
        if len(self.result.multi_hand_landmarks) > 0 and hand_number in self.result.multi_hand_landmarks:
            for id, land_mark in enumerate(self.result.multi_hand_landmarks[hand_number]):
                position_dict[id] = (int(land_mark.x * w), int(land_mark.y * h))
                if draw:
                    self.draw.draw_landmarks(image, land_mark, self.solutions.HAND_CONNECTIONS)
        return position_dict, image
