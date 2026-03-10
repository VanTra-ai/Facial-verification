import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. TẢI DỮ LIỆU ĐẶC TRƯNG
# ==========================================
data_path = 'data_face_features.pickle'
print("----- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH -----")

with open(data_path, 'rb') as f:
    data = pickle.load(f)

X = data['data']
y = data['labels']

danh_sach_ten = list(set(y))
print(f"[*] Đã tải {len(X)} mẫu dữ liệu từ {len(danh_sach_ten)} người.")
print(f"[*] Danh sách nhận diện: {danh_sach_ten}")

# ==========================================
# 2. KHỞI TẠO THUẬT TOÁN (MACHINE LEARNING)
# ==========================================
# Thuật toán 1: Support Vector Machine (Nhớ bật probability=True)
svm_model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)

# Thuật toán 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

# Kết hợp sức mạnh của cả 2 (Bắt buộc dùng voting='soft' để lấy phần trăm tự tin sau này)
voting_model = VotingClassifier(
    estimators=[('svm', svm_model), ('rf', rf_model)], 
    voting='soft'
)

# ==========================================
# 3. TIẾN HÀNH HUẤN LUYỆN (TRAINING)
# ==========================================
print("\n[*] Đang tiến hành cho AI học... Vui lòng chờ...")
voting_model.fit(X, y)

# Kiểm tra xem AI đã học thuộc bài tốt chưa
y_pred = voting_model.predict(X)
do_chinh_xac = accuracy_score(y, y_pred)
print(f"[*] Độ chính xác trên tập dữ liệu hiện tại: {do_chinh_xac * 100:.2f}%")

# ==========================================
# 4. LƯU LẠI MÔ HÌNH SAU KHI HỌC
# ==========================================
model_output_path = 'ml_face_person_identity.pkl'
with open(model_output_path, 'wb') as f:
    pickle.dump(voting_model, f)

print(f"\n[THÀNH CÔNG] Đã lưu bộ não nhận diện vào file: {model_output_path}")