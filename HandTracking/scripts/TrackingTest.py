import HandTracker as ht
import cv2
import time

video = cv2.VideoCapture(0)
Hand = ht.HandTracker()
start, end = 0,0

while True:
    success, image = video.read()    
    positions, image = Hand.findPosition(image, hand_number=0, draw=True)
    if positions:
        print(positions[4])
    ## calculating the fps
    end = time.time()
    fps = 1 / (end - start)
    start = end
    cv2.putText(image, str(int(fps)), (10,70),
                cv2.FONT_HERSHEY_PLAIN, 3,
                (0,0,0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(1)

