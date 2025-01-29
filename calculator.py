import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, is_left_hand):
    
    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    
    if is_left_hand:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            count += 1
    else:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1

    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    
    return count

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        left_hand_value = 0
        right_hand_value = 0

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_label = results.multi_handedness[idx].classification[0].label
                if hand_label == "Left":
                    left_hand_value = count_fingers(hand_landmarks, is_left_hand=True)
                elif hand_label == "Right":
                    right_hand_value = count_fingers(hand_landmarks, is_left_hand=False)

        total_fingers = left_hand_value + right_hand_value

        cv2.putText(frame, f"Tangan Kiri: {left_hand_value}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Tangan Kanan: {right_hand_value}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Jari: {total_fingers}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Calculator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
