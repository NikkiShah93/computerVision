import HandTracker as ht
import cv2

video = cv2.VideoCapture(0)

while True:
    success, image = video.read()
    Hand = ht.HandTracker()
    positions, image = Hand.findPosition(image, hand_number=0)
    if positions:
        print(positions[4])
    
    cv2.imshow("Image", image)
    cv2.waitKey(1)

