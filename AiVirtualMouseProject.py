# import cv2
# import numpy as np
# import time
# from module import HandTrackingModule as htm




# ##############################
# wCam, hCam = 640, 480
# ##############################

# cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
# pTime = 0
# detector = htm.handDetector(maxHands=1)


# while True:
#     # 1. Finding hand Landmarks
#     success, img = cap.read()
#     img = detector.findHands(img)
#     lmList, bbox = detector.findPosition(img)
        
#     # 2. Get the tip of the index and middle fingers
#     # 3. Check which fingers are up
#     # 4. Only Index Finger : Moving Mode
#         # 5. Convert Coordinates
#     # 6. Smoothen Values
#     # 7. Move Mouse
#     # 8. Both Index and middle fingers are up : Clicking Mode
#     # 9. Find distance between fingers
#     # 10. Click mouse if distance short
    
#     # 11. Frame Rate
#     cTime =time.time()
#     fps = 1/(cTime-pTime)
#     pTime = cTime
#     cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                 (255, 0, 0), 3)
#     # 12. Display
#     cv2.imshow("Image", img)
    
#     cv2.waitKey(1)

import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
from module import HandTrackingModule as htm  # HandTrackingModule을 htm으로 import

# 웹캠 설정
wCam, hCam = 640, 480
frameR = 100  # 프레임 감소
smoothening = 7

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.HandDetector(maxHands=1)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # 검지 끝 좌표
        x2, y2 = lmList[12][1:]  # 중지 끝 좌표

        # 손가락이 위로 향했는지 확인
        fingers = detector.fingers_up()
        if fingers[1] == 1 and fingers[2] == 0:
            # 화면 크기에 맞게 좌표 변환
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 부드럽게 마우스 움직임 처리
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 마우스 이동
            pyautogui.moveTo(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

        # 클릭 모드
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.find_distance(8, 12, img)
            if length < 40:
                pyautogui.click()

    # FPS 계산
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
