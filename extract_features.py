import cv2
import numpy as np
import os
import pickle

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & LOAD MÔ HÌNH
# ==========================================
faceDetectionModel = './models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
faceDetectionProto = './models/deploy.prototxt.txt'
faceDescriptor = './models/openface.nn4.small2.v1.t7'

# Load AI tìm khuôn mặt và AI trích xuất đặc trưng
detectorModel = cv2.dnn.readNetFromCaffe(faceDetectionProto, faceDetectionModel)
descriptorModel = cv2.dnn.readNetFromTorch(faceDescriptor)

dataset_path = './faces'
face_data = [] # Danh sách chứa các vector 128 số
labels = []    # Danh sách chứa tên tương ứng (VD: 'PhamVanTra')

print("----- BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG -----")

# ==========================================
# 2. QUÉT THƯ MỤC VÀ XỬ LÝ ẢNH
# ==========================================
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    # Bỏ qua nếu không phải là thư mục
    if not os.path.isdir(person_folder): 
        continue

    print(f">> Đang xử lý dữ liệu của: {person_name}")
    
    for image_name in os.listdir(person_folder):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')): 
            continue
            
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)
        if img is None: 
            continue

        h, w = img.shape[:2]
        
        # Tìm khuôn mặt trong ảnh
        img_blob = cv2.dnn.blobFromImage(img, 1, (300,300), (104,177,123), swapRB=False, crop=False)
        detectorModel.setInput(img_blob)
        detections = detectorModel.forward()

        if len(detections) > 0:
            # Lấy khuôn mặt có độ tự tin cao nhất
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startx, starty, endx, endy = box.astype(int)
                
                # Cắt lấy khuôn mặt
                face_roi = img[max(0, starty):min(h, endy), max(0, startx):min(w, endx)]
                if face_roi.shape[0] < 10 or face_roi.shape[1] < 10: 
                    continue

                # Đưa khuôn mặt qua OpenFace để lấy vector 128 chiều
                face_blob = cv2.dnn.blobFromImage(face_roi, 1/255.0, (96, 96), (0, 0, 0), swapRB=True, crop=True)
                descriptorModel.setInput(face_blob)
                vectors = descriptorModel.forward()

                # Lưu dữ liệu vào danh sách
                face_data.append(vectors[0])
                labels.append(person_name)

# ==========================================
# 3. LƯU KẾT QUẢ RA FILE PICKLE
# ==========================================
data = {'data': face_data, 'labels': labels}
output_file = 'data_face_features.pickle'

with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print(f"\n[THÀNH CÔNG] Đã trích xuất xong {len(face_data)} khuôn mặt!")
print(f"Dữ liệu đã được gói gọn vào file: {output_file}")