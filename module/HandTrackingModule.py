import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingers_up(self):
        fingers = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            # 엄지
            if myHand.landmark[4].x < myHand.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
            # 나머지 네 손가락
            for id in range(1, 5):
                if myHand.landmark[4 * id + 4].y < myHand.landmark[4 * id + 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.results.multi_hand_landmarks[0].landmark[p1].x, self.results.multi_hand_landmarks[0].landmark[p1].y
        x2, y2 = self.results.multi_hand_landmarks[0].landmark[p2].x, self.results.multi_hand_landmarks[0].landmark[p2].y
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if draw:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
            cv2.circle(img, (int(x1), int(y1)), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (int(x2), int(y2)), 10, (0, 255, 0), cv2.FILLED)
        return length, img, [(x1, y1), (x2, y2)]
