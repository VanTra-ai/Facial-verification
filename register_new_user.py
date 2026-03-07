import cv2
import numpy as np
import os
import time
import sys
import mediapipe as mp

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & MEDIA PIPE
# ==========================================
faceDetectionModel = './models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
faceDetectionProto = './models/deploy.prototxt.txt'
detectorModel = cv2.dnn.readNetFromCaffe(faceDetectionProto, faceDetectionModel)

dataset_path = './faces'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hàm tính toán góc quay của đầu (Head Pose)
def get_head_pose(image):
    img_h, img_w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None, None
        
    for face_landmarks in results.multi_face_landmarks:
        # Lấy tọa độ 6 điểm chuẩn trên mặt
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [1, 33, 263, 61, 291, 152]: # Mũi, 2 Mắt, 2 Mép, Cằm
                if idx == 1: nose_2d = (lm.x * img_w, lm.y * img_h)
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
                
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Ma trận camera mô phỏng
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Tính toán góc
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # angles[0] = Pitch (lên/xuống), angles[1] = Yaw (trái/phải), angles[2] = Roll (nghiêng vai)
        pitch = angles[0] * 360
        yaw = angles[1] * 360
        
        return pitch, yaw
    return None, None

# ==========================================
# 2. NHẬP THÔNG TIN NGƯỜI MỚI
# ==========================================
user_name = input("Nhap Ten hoac ID nguoi dung moi (VD: van_tra_01): ")
user_folder = os.path.join(dataset_path, user_name)

if not os.path.exists(user_folder):
    os.makedirs(user_folder)
    print(f"Da tao thu muc luu tru: {user_folder}")
else:
    print(f"Thu muc {user_folder} da ton tai.")

# ==========================================
# 3. KỊCH BẢN CHỤP ẢNH (Giao diện eKYC + Check Góc)
# ==========================================
poses = [
    {"name": "CHINH DIEN", "count": 15},
    {"name": "NGHIENG TRAI", "count": 10},
    {"name": "NGHIENG PHAI", "count": 10}
]

cap = cv2.VideoCapture(0)
GUIDE_W, GUIDE_H = 220, 300

for pose in poses:
    pose_name = pose["name"]
    target_count = pose["count"]
    captured_count = 0
    is_recording = False 
    
    while captured_count < target_count:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        image = frame.copy()
        h, w = frame.shape[:2]
        
        gx1, gy1 = (w - GUIDE_W) // 2, (h - GUIDE_H) // 2
        gx2, gy2 = gx1 + GUIDE_W, gy1 + GUIDE_H
        
        guide_color = (255, 255, 255)
        message = "Khong thay khuon mat"
        is_valid_position = False
        
        # 1. Tìm khuôn mặt bằng Caffe Model (Như cũ)
        img_blob = cv2.dnn.blobFromImage(frame, 1, (300,300), (104,177,123), swapRB=False, crop=False)
        detectorModel.setInput(img_blob)
        detections = detectorModel.forward()
        
        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startx, starty, endx, endy = box.astype(int)
                cv2.rectangle(image, (startx, starty), (endx, endy), (200, 200, 200), 1)
                
                face_w = endx - startx
                face_center_x, face_center_y = startx + face_w // 2, starty + (endy - starty) // 2
                guide_center_x, guide_center_y = w // 2, h // 2
                
                # Check Vị trí & Khoảng cách
                if abs(face_center_x - guide_center_x) > 40 or abs(face_center_y - guide_center_y) > 40:
                    message = "Vui long dua mat vao GIUA khung!"
                    guide_color = (0, 0, 255)
                elif face_w < GUIDE_W * 0.55:
                    message = "Tien lai GAN hon mot chut!"
                    guide_color = (0, 165, 255)
                elif face_w > GUIDE_W * 0.95:
                    message = "Lui ra XA hon mot chut!"
                    guide_color = (0, 165, 255)
                else:
                    # 2. CHECK GÓC MẶT BẰNG MEDIAPIPE
                    pitch, yaw = get_head_pose(image)
                    
                    if pitch is not None and yaw is not None:
                        # Ràng buộc góc tùy theo thao tác
                        if pose_name == "CHINH DIEN":
                            if abs(yaw) > 15 or abs(pitch) > 15: # Sai lệch quá 15 độ là báo lỗi
                                message = "Loi: Vui long NHIN THANG!"
                                guide_color = (0, 0, 255)
                            else:
                                message = "Chinh dien OK! Bam 'c' de chup"
                                guide_color = (0, 255, 0)
                                is_valid_position = True
                        
                        elif pose_name == "NGHIENG TRAI":
                            if yaw > -15: # Cần quay trái đủ sâu (âm)
                                message = "Loi: Vui long QUAY MAT SANG TRAI!"
                                guide_color = (0, 0, 255)
                            else:
                                message = "Trai OK! Bam 'c' de chup"
                                guide_color = (0, 255, 0)
                                is_valid_position = True
                                
                        elif pose_name == "NGHIENG PHAI":
                            if yaw < 15: # Cần quay phải đủ sâu (dương)
                                message = "Loi: Vui long QUAY MAT SANG PHAI!"
                                guide_color = (0, 0, 255)
                            else:
                                message = "Phai OK! Bam 'c' de chup"
                                guide_color = (0, 255, 0)
                                is_valid_position = True

        # Vẽ giao diện
        cv2.rectangle(image, (gx1, gy1), (gx2, gy2), guide_color, 2)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, gy1), (0,0,0), -1)
        cv2.rectangle(overlay, (0, gy2), (w, h), (0,0,0), -1)
        cv2.rectangle(overlay, (0, gy1), (gx1, gy2), (0,0,0), -1)
        cv2.rectangle(overlay, (gx2, gy1), (w, gy2), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        if not is_recording:
            cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)
        else:
            if is_valid_position:
                status = f"Dang chup {pose_name}: {captured_count}/{target_count}"
                cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                timestamp = int(time.time() * 1000)
                file_name = f"{user_name}_{pose_name.replace(' ', '_')}_{timestamp}.jpg"
                file_path = os.path.join(user_folder, file_name)
                
                original_frame = cv2.flip(frame, 1) 
                cv2.imwrite(file_path, original_frame)
                
                captured_count += 1
                time.sleep(0.1) 
            else:
                cv2.putText(image, "SAI THAO TAC! Vui long lam lai", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('eKYC - Dang ky khuon mat', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not is_recording and is_valid_position:
            is_recording = True 
        elif key == ord('q'):
            print("Đã hủy.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"-> Hoan thanh {target_count} anh cho goc {pose_name}!")
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print(f"\n[THANH CONG] Da luu anh vao {user_folder}")