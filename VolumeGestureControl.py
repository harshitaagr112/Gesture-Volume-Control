import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam = 640
hCam = 480


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPercentage = 0
area = 0
volumeColor = (255,0,0)

cap = cv.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detection_confidence=0.7, max_hands=2)

pTime = 0
while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    img = detector.findHands(img)
    landmarkList, bounding_box = detector.findPosition(img,draw = True, color = (255,0,0))

    if len(landmarkList) !=0:
        area = (bounding_box[2]-bounding_box[0])*(bounding_box[3]-bounding_box[1])//100

        if 250<area<1000:
            length, img, line_info = detector.findDistance(4,8,img)
            print(length)

            volBar = np.interp(length, [50, 220], [400,150])
            volPercentage = np.interp(length, [50, 220], [0,100])
            smoothness = 10
            volPercentage = smoothness * round(volPercentage/smoothness)
            fingers = detector.fingersUp()
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPercentage/100, None)
                cv.circle(img, (line_info[4], line_info[5]), 10, (0, 255, 0), cv.FILLED)
                volumeColor = (0,255,0)
            else:
                volumeColor = (255,0,0)

    cv.rectangle(img, (50,150), (85,400), (255,0,0),3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255,0,0), cv.FILLED)
    cv.putText(img, str(int(volPercentage))+"%", (40,450), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cVol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv.putText(img, "Vol. Set : "+str(cVol) + "%", (400, 50), cv.FONT_HERSHEY_COMPLEX, 0.75, volumeColor, 2)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # img = cv.flip(img, 1)
    cv.putText(img, "FPS: "+str(int(fps)), (30, 50), cv.FONT_HERSHEY_COMPLEX, 0.75, (225, 0, 0), 2)
    cv.imshow("Hand Tracking", img)
    cv.waitKey(1)