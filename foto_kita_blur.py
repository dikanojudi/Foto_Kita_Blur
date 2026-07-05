import cv2
import mediapipe as mp
import numpy as np
import pygame

pygame.mixer.init()
pygame.mixer.music.load("lagu.mp3") 

musik_sedang_bermain = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gestur_terpenuhi = False
    frame_asli = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            telunjuk_pip = hand_landmarks.landmark[6].y
            tengah_pip = hand_landmarks.landmark[10].y
            manis_pip = hand_landmarks.landmark[14].y
            kelingking_pip = hand_landmarks.landmark[18].y

            telunjuk_terbuka = hand_landmarks.landmark[8].y < telunjuk_pip
            tengah_terbuka = hand_landmarks.landmark[12].y < tengah_pip
            manis_terbuka = hand_landmarks.landmark[16].y < manis_pip
            kelingking_terbuka = hand_landmarks.landmark[20].y < kelingking_pip
            
            if telunjuk_terbuka and tengah_terbuka and (not manis_terbuka) and (not kelingking_terbuka):
                gestur_terpenuhi = True
                
                mask = np.zeros_like(frame_asli, dtype=np.uint8)
                mp_drawing.draw_landmarks(mask, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                frame_blured_full = cv2.GaussianBlur(frame_asli, (35, 35), 0)
                tangan_tajam = cv2.bitwise_and(frame_asli, frame_asli, mask=mask)
                mask_inv = cv2.bitwise_not(mask)
                latar_blur = cv2.bitwise_and(frame_blured_full, frame_blured_full, mask=mask_inv)
                
                frame = cv2.add(tangan_tajam, latar_blur)

                cv2.putText(frame, "Foto Kita Blur!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if gestur_terpenuhi:
        if not musik_sedang_bermain:
            pygame.mixer.music.play(-1) 
            musik_sedang_bermain = True
    else:
        if musik_sedang_bermain:
            pygame.mixer.music.stop()   
            musik_sedang_bermain = False

    cv2.imshow('Hand Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
