import cv2 
import numpy as np
import time
import os


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    print(key)
    if key ==32:
        cv2.imwrite("C:/Edward/Sign.Language.Detection/temporary.jpg", frame)
        os.system('python3 predict.py temporary.jpg')
        time.sleep(5)
cap.release()
cv2.destroyAllWindows()
