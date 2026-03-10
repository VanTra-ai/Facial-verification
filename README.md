# 🧑‍💻 Hệ Thống Nhận Diện & Điểm Danh Khuôn Mặt (Facial Verification System)

Đây là dự án phát triển hệ thống điểm danh tự động dựa trên công nghệ nhận diện khuôn mặt (Facial Recognition). Điểm nổi bật của dự án là việc tích hợp **công nghệ eKYC (Định danh điện tử)** trong bước thu thập dữ liệu, giúp ép buộc người dùng thực hiện đúng các thao tác góc mặt, ngăn chặn việc thu thập dữ liệu rác và nâng cao độ chính xác của mô hình.

## ✨ Tính Năng Nổi Bật

* **Thu thập dữ liệu eKYC (Anti-Spoofing & Pose Estimation):** Ứng dụng `MediaPipe` để lập bản đồ 468 điểm 3D trên khuôn mặt, tính toán góc nghiêng (Yaw, Pitch, Roll) theo thời gian thực để hướng dẫn người dùng nhìn thẳng, quay trái, quay phải.
* **Trích xuất đặc trưng chuẩn xác:** Sử dụng mô hình Deep Learning **OpenFace** (`openface.nn4.small2.v1.t7`) để chuyển đổi hình ảnh khuôn mặt thành các vector 128 chiều.
* **Mô hình Phân loại Kép (Ensemble Learning):** Kết hợp sức mạnh của `Support Vector Machine (SVM)` và `Random Forest` thông qua cơ chế *Soft Voting* để đạt độ chính xác tối đa.
* **Nhận diện Thời gian thực (Real-time):** Quét và nhận diện khuôn mặt trực tiếp qua Camera. Tích hợp ngưỡng tự tin (Threshold) để phát hiện và gắn nhãn "Người Lạ" (Unknown) đối với những người chưa đăng ký.

## 📁 Cấu Trúc Dự Án

* `register_new_user.py`: Script khởi động Camera với khung hướng dẫn eKYC để chụp ảnh người dùng mới và lưu vào thư mục `./faces/`.
* `extract_features.py`: Quét toàn bộ ảnh trong thư mục `./faces/`, tìm khuôn mặt và chuyển đổi thành vector đặc trưng (lưu ra file `data_face_features.pickle`).
* `train_model.py`: Load dữ liệu vector để huấn luyện mô hình Machine Learning và xuất ra "bộ não" `ml_face_person_identity.pkl`.
* `live_recognition.py`: Script chạy hệ thống điểm danh trực tiếp qua Webcam.
* `models/`: Thư mục chứa các mạng Nơ-ron (Caffe, Torch) dùng để phát hiện mặt và trích xuất đặc trưng.

## ⚙️ Yêu Cầu Môi Trường (Prerequisites)

Để hệ thống hoạt động trơn tru và không bị xung đột lõi C++, hệ thống yêu cầu cấu hình môi trường chuẩn như sau:
* Python 3.10
* numpy == 1.26.4
* mediapipe == 0.10.14
* protobuf == 3.20.3
* opencv-contrib-python

**Lệnh cài đặt thư viện:**
```bash
pip install numpy==1.26.4 mediapipe==0.10.14 protobuf==3.20.3 opencv-contrib-python scikit-learn
```

## 🚀 Hướng Dẫn Chạy

Hệ thống được vận hành theo quy trình 4 bước chuẩn mực:

**Bước 1: Đăng ký người dùng mới**
```bash
python register_new_user.py
```
Nhập tên người dùng (VD: PhamVanTra). Đưa mặt vào khung xanh giữa màn hình và thực hiện các động tác Nhìn thẳng, Quay Trái/Phải theo hướng dẫn của hệ thống eKYC để chụp ảnh

**Bước 2: Trích xuất đặc trưng khuôn mặt**
```bash
python extract_features.py
```
Hệ thống sẽ chuyển hóa các bức ảnh vừa chụp thành file dữ liệu số **data_face_features.pickle**.

**Bước 3: Huấn luyện bộ máy AI (Training)**
```bash
python train_model.py
```
Huấn luyện mô hình SVM và Random Forest. Kết quả lưu tại **ml_face_person_identity.pkl**.

**Bước 4: Khởi động hệ thống Điểm Danh**
```bash
python live_recognition.py
```
Bật Camera, đứng vào khung hình và xem kết quả nhận diện!
