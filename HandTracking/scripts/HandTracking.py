import numpy as np
import cv2
import mediapipe as mp
import time

## tracking the time
past_time = 0
curr_time = 0

## capturing the webcam 
video = cv2.VideoCapture(0)
## creating an instance of the mp hand
## with the default values for now
mphand = mp.solutions.hands
hands = mphand.Hands()
## we need the drawing from mp
## to draw the points on the detected hands
drawhands = mp.solutions.drawing_utils
while True:
    success, img = video.read()
    ## we need to change the colors for the captured video
    rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ## and then process the info
    results = hands.process(rbg_img)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            drawhands.draw_landmarks(img, hand, mphand.HAND_CONNECTIONS)
            for id, lm in enumerate(hand.landmarks):
                print(id, lm)
    ## calculating the fps
    curr_time = time.time()
    fps = 1 / (curr_time - past_time)
    past_time = curr_time
    ## and the adding it to the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL,3,
                color=(0, 0, 0), thickness=1)
    
    ## showing the captured video
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# When everything done, release the capture
# video.release()
# cv2.destroyAllWindows()