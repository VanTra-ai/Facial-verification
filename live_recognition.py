import cv2
import numpy as np
import pickle
import time

# ==========================================
# 1. LOAD TẤT CẢ CÁC MÔ HÌNH VÀ "BỘ NÃO"
# ==========================================
print("[-] Đang tải các mô hình AI... Vui lòng chờ...")
faceDetectionModel = './models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
faceDetectionProto = './models/deploy.prototxt.txt'
faceDescriptor = './models/openface.nn4.small2.v1.t7'
ml_model_path = './ml_face_person_identity.pkl'

# Load AI tìm khuôn mặt (Caffe) và AI trích đặc trưng (Torch)
detectorModel = cv2.dnn.readNetFromCaffe(faceDetectionProto, faceDetectionModel)
descriptorModel = cv2.dnn.readNetFromTorch(faceDescriptor)

# Load "Bộ não" Machine Learning đã train
with open(ml_model_path, 'rb') as f:
    face_recognition_model = pickle.load(f)

print("[-] Tải mô hình thành công! Đang bật Camera...")

# ==========================================
# 2. KHỞI ĐỘNG CAMERA & NHẬN DIỆN THỜI GIAN THỰC
# ==========================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Lật camera giống soi gương cho dễ nhìn
    frame = cv2.flip(frame, 1)
    image = frame.copy()
    h, w = image.shape[:2]

    # Bước 1: Phát hiện khuôn mặt
    img_blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)
    detectorModel.setInput(img_blob)
    detections = detectorModel.forward()

    if len(detections) > 0:
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Chỉ xử lý những khuôn mặt rõ ràng (độ tự tin > 60%)
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startx, starty, endx, endy = box.astype(int)
                
                # Cắt lấy khuôn mặt
                face_roi = image[max(0, starty):min(h, endy), max(0, startx):min(w, endx)]
                if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                    continue
                    
                # Bước 2: Rút trích đặc trưng 128 số bằng OpenFace
                face_blob = cv2.dnn.blobFromImage(face_roi, 1/255.0, (96, 96), (0, 0, 0), swapRB=True, crop=True)
                descriptorModel.setInput(face_blob)
                vectors = descriptorModel.forward()
                
                # Bước 3: Dùng Bộ Não dự đoán xem đây là ai
                face_name = face_recognition_model.predict(vectors)[0]
                face_score = face_recognition_model.predict_proba(vectors).max()
                
                # BƯỚC 4: HIỂN THỊ KẾT QUẢ VÀ TẠO NGƯỠNG AN TOÀN (THRESHOLD)
                # Nếu độ tự tin dưới 60%, coi như là người lạ (Unknown)
                if face_score > 0.60:
                    color = (0, 255, 0) # Xanh lá cho người quen
                    display_name = face_name
                else:
                    color = (0, 0, 255) # Đỏ cho người lạ
                    display_name = "Nguoi La (Unknown)"

                text = f"{display_name}: {face_score*100:.0f}%"
                
                # Vẽ khung vuông
                cv2.rectangle(image, (startx, starty), (endx, endy), color, 2)
                # Vẽ nền chữ nhật cho text dễ nhìn
                cv2.rectangle(image, (startx, starty - 25), (endx, starty), color, -1)
                # Viết tên lên màn hình
                cv2.putText(image, text, (startx + 5, starty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('He Thong Nhan Dien Khuon Mat (Live)', image)
    
    # Bấm phím 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Đã tắt Camera và kết thúc chương trình.")