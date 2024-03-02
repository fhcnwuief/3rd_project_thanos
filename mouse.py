# mouse연결 ver0.8_4

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import subprocess       # virtual keyboard 나타낼 때 필요 
import os               # virtual keyboard 닫을 때 필요         
import pygetwindow as gw
import psutil
import time

pyautogui.FAILSAFE = False

# 다른 코드를 그대로 유지하면서 Controller.flag를 추가합니다.
class Controller:
    flag = False

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

# Define the screen resolution (you may adjust this based on your monitor)
screen_width, screen_height = pyautogui.size()   # 추가
print(screen_width, screen_height)

# Gesture recognition model
file = np.genfromtxt('./data/gesture_train_fy.csv', delimiter=',')
# file = np.genfromtxt('./BB_son/data/gesture_train_fy.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
next_cnt = 0        # 수정 변수초기화  동일한 gesture count


# 실행할 vkeyboard.pyw 파일 경로
pyw_file_path = './vkeyboard.pyw'       # 경로수정 해주세요
# pyw_file_path = './vkey_main/vkeyboard.pyw'       # 경로수정 해주세요
try:
    # .pyw 파일 실행
    subprocess.Popen(['pythonw', pyw_file_path])
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")   # 예외처리
except Exception as e:
    print(f"오류 발생: {e}")


while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        continue
 
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = img.shape

    # 카메라 화면 크기 가져오기
    camera_x = int(cap.get(3))
    camera_y = int(cap.get(4))
    
    # 사각형 그리기 (예: 화면 중앙에 사각형 그리기)
    # rect_width = 480
    # rect_height = 360
    rect_width = 1048    # macbook
    rect_height = 480    # macbook
    rect_x = (camera_x - rect_width) // 2
    rect_y = (camera_y - rect_height) // 2
    rect_right = rect_x + rect_width
    rect_bottom = rect_y + rect_height
    
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 0), 2)

    # 프레임 표시
    #cv2.imshow("Camera Feed", img)  


    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # fy동작일때 모자이크 처리되면서 esc기능
            if idx == 11:
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img
                
                
                pyautogui.press('esc')
                

            
            # rock일때 마우스 드래그
            if idx == 0:
                #print('mouse drag')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
               
                # 손목좌표 입력
                wrist_x, wrist_y = tuple((res.landmark[mp_hands.HandLandmark.WRIST].x * image_width, res.landmark[mp_hands.HandLandmark.WRIST].y * image_height))
                # Map the fingertip coordinates to the screen resolution
                # 손목좌표를 사각형 내부의 상대좌표로 변환
                relative_x = wrist_x - rect_x
                relative_y = wrist_y - rect_y
                # 상대좌표를 전체화면 크기에 비례하여 조정
                screen_x = int(relative_x * screen_width / rect_width)
                screen_y = int(relative_y * screen_height / rect_height)
                # 마우스 좌표 설정
                position_x = rect_x + screen_x
                position_y = rect_y + screen_y
                
           
                if next_cnt > 2:        #똑같은 동작이 5번 인식되어야 실행
                    next_cnt = 0
                    if not Controller.flag:
                        Controller.flag = True
                        pyautogui.mouseDown(button='left', x=position_x, y=position_y)
                    else:
                        pyautogui.moveTo(position_x, position_y, duration=0.1)  # Adjust the values as needed
                   
                else:
                    continue

            # paper일때 마우스 left드롭
            if idx == 5:
                #print('mouse left click')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
                
                # 손목좌표 입력
                wrist_x, wrist_y = tuple((res.landmark[mp_hands.HandLandmark.WRIST].x * image_width, res.landmark[mp_hands.HandLandmark.WRIST].y * image_height))
                # Map the fingertip coordinates to the screen resolution
                # 손목좌표를 사각형 내부의 상대좌표로 변환
                relative_x = wrist_x - rect_x
                relative_y = wrist_y - rect_y
                # 상대좌표를 전체화면 크기에 비례하여 조정
                screen_x = int(relative_x * screen_width / rect_width)
                screen_y = int(relative_y * screen_height / rect_height)
                # 마우스 좌표 설정
                position_x = rect_x + screen_x
                position_y = rect_y + screen_y
            
                if next_cnt > 2:        #똑같은 동작이 5번 인식되어야 실행
                    next_cnt = 0    
                    if Controller.flag:
                        Controller.flag = False
                        pyautogui.mouseUp(button='left')
                        pyautogui.moveTo(position_x, position_y, duration=0.1)
                    else :
                        pyautogui.moveTo(position_x, position_y, duration=0.1)
                else:
                    continue
                    

                
            # 8은 'spiderman'에 해당하는 제스처입니다.           
            if idx == 8:  
                #print('page refresh')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
                if next_cnt > 4:        #똑같은 동작이 3번 인식되어야 실행
                    next_cnt = 0
                    pyautogui.press('F5')
                else:
                    continue
                
            if idx == 10:  # 10은 'okay'에 해당하는 제스처입니다.
                #print('mouse doubleclick')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
                if next_cnt > 4:        #똑같은 동작이 5번 인식되어야 실행
                    next_cnt = 0 
                    pyautogui.doubleClick()
                else:
                    continue
                    
            if idx == 9:  # 9은 'yeah=V'에 해당하는 제스처입니다.
                #print('right click')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
                if next_cnt > 4:        #똑같은 동작이 5번 인식되어야 실행
                    next_cnt = 0 
                    pyautogui.rightClick()
                else:
                    continue
                    
                    
            if idx == 4:  # 4은 '4'에 해당하는 제스처입니다.
                #print('wind+d')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
                if next_cnt > 15:        #똑같은 동작이 5번 인식되어야 실행
                    next_cnt = 0 
                    pyautogui.hotkey('win','d')
                else:
                    continue
                    
                    
            if idx == 6:  # 6은 🤙에 해당하는 제스처입니다.
                #print('backspace')
                next_cnt += 1
                #print('next_cnt: ', next_cnt)
                if next_cnt > 4:        #똑같은 동작이 5번 인식되어야 실행
                    next_cnt = 0 
                    pyautogui.press('backspace')
                else:
                    continue
                 
    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == 27:   # q->esc
        break


    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
# click(botton='right')