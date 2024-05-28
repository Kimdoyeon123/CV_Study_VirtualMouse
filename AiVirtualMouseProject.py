import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui



##############################
wCam, hCam = 640, 480
##############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중앙 점
    c = np.array(c)  # 마지막 점
    
    # 벡터 계산
    ba = a - b
    bc = c - b
    
    # 코사인 법칙을 사용하여 각도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


# MediaPipe hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


# Mediapipe Hands 솔루션 사용
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 이미지를 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손가락별로 구분하여 각 랜드마크 좌표 출력
                for id, lm in enumerate(hand_landmarks.landmark):
                    # 랜드마크 좌표를 이미지 크기에 맞게 조정
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # 각 손가락의 각도 계산 (엄지손가락을 제외한 네 손가락)
                    if id == 8:
                        # 손가락 끝 (Tip)
                        tip = [hand_landmarks.landmark[id].x, hand_landmarks.landmark[id].y]
                        # 손가락 중간 마디 (PIP)
                        pip = [hand_landmarks.landmark[id - 2].x, hand_landmarks.landmark[id - 2].y]
                        # 손가락 기저부 (MCP)
                        mcp = [hand_landmarks.landmark[id - 3].x, hand_landmarks.landmark[id - 3].y]

                        angle = calculate_angle(mcp, pip, tip)

                        # 각도 출력 (예를 들어 160도 이상이면 손가락을 폈다고 간주)
                        if id == 8 and angle > 160:
                            pyautogui.position(cx, cy)
                            #print(f"Finger {id // 4} is straight")
                            
                        elif id == 8 and angle < 160:
                            pyautogui.click(cx, cy)
                            #print(f"Finger {id // 4} is bent")
                    
                    
                    
                    # # 각 손가락에 해당하는 랜드마크를 출력
                    # if id == 4:
                        
                    #     print("Thumb Tip: ", cx, cy)
                    # elif id == 8:
                    #     pyautogui.click(cx, cy)
                    #     print("Index Finger Tip: ", cx, cy)
                    # elif id == 12:
                    #     print("Middle Finger Tip: ", cx, cy)
                    # elif id == 16:
                    #     print("Ring Finger Tip: ", cx, cy)
                    # elif id == 20:
                    #     print("Pinky Finger Tip: ", cx, cy)
        cTime =time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 12. Display
        cv2.imshow("Image", image)
    
        if cv2.waitKey(1) & 0xFF == 27:
            break    
    # 2. Get the tip of the index and middle fingers
    # 3. Check which fingers are up
    # 4. Only Index Finger : Moving Mode
        # 5. Convert Coordinates
    # 6. Smoothen Values
    # 7. Move Mouse
    # 8. Both Index and middle fingers are up : Clicking Mode
    # 9. Find distance between fingers
    # 10. Click mouse if distance short
    
    # 11. Frame Rate
   
