import cv2
import mediapipe as mp

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

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
                    
                    # 각 손가락에 해당하는 랜드마크를 출력
                    if id == 4:
                        print("Thumb Tip: ", cx, cy)
                    elif id == 8:
                        print("Index Finger Tip: ", cx, cy)
                    elif id == 12:
                        print("Middle Finger Tip: ", cx, cy)
                    elif id == 16:
                        print("Ring Finger Tip: ", cx, cy)
                    elif id == 20:
                        print("Pinky Finger Tip: ", cx, cy)

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
