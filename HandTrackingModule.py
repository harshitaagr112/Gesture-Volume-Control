import cv2 as cv
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNumber = 0, draw = True, color = (255,0,0)):
        xlist = []
        ylist = []
        bounding_box = []
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.landmark_list.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx,cy), 5, color , cv.FILLED)

            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bounding_box = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(img, (bounding_box[0]-20,bounding_box[1]-20),
                            (bounding_box[2]+20,bounding_box[3]+20), (0,255,0), 2)

        return self.landmark_list,bounding_box


    def fingersUp(self):
        fingers = []
        self.tipIds = [4, 8, 12, 16, 20]
        # Thumb
        if self.landmark_list[self.tipIds[0]][1] < self.landmark_list[self.tipIds[0] - 1 ][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1,5):
            if self.landmark_list[self.tipIds[id]][2] < self.landmark_list[self.tipIds[id] - 2 ][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self,p1,p2,img,draw = True):
        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


        if draw:
            cv.circle(img, (x1, y1), 10, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 10, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1,y1,x2,y2,cx,cy]









def main():
    cap = cv.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        img = detector.findHands(img)
        landmarkList, building_box = detector.findPosition(img)
        if len(landmarkList) !=0:
            print(landmarkList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # img = cv.flip(img, 1)
        cv.putText(img, str(int(fps)), (30, 50), cv.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)
        cv.imshow("Hand Tracking", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
