## 🏷️ Quá trình đánh nhãn dữ liệu

Tập dữ liệu trong dự án **Trash Classification** được **đánh nhãn thủ công bằng LabelImg** ở **định dạng YOLO**.  
Mỗi ảnh có tệp nhãn `.txt` tương ứng, trong đó mỗi dòng mô tả một đối tượng theo cấu trúc:

<class_id> <x_center> <y_center> <width> <height>

Trong đó các giá trị tọa độ được **chuẩn hóa (normalized)** theo kích thước ảnh.  
Bộ dữ liệu gồm **3 nhãn chính**:

- 📰 `paper`  
- 🧴 `plastic`  
- 🍾 `glass`

Tập nhãn này được sử dụng cho các mô hình huấn luyện theo chuẩn **YOLOv5 / YOLOv8**.
