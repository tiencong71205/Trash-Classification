import onnxruntime as ort
import cv2
import numpy as np
import time

# ====== Cấu hình ======
model_path = "3.refine_last_phase3_10epochs.onnx"
img_path = "test.jpg"
class_names = ["paper", "plastic", "glass"]
img_size = 640
conf_thresh = 0.3
iou_thresh = 0.45

# ====== Load model ======
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ====== Load & tiền xử lý ảnh ======
img0 = cv2.imread(img_path)
if img0 is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh {img_path}")

img = cv2.resize(img0, (img_size, img_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, 0).astype(np.float32) / 255.0

# ====== Suy luận ======
start = time.time()
preds = session.run([output_name], {input_name: img})[0]
fps = 1 / (time.time() - start)
print(f"⏱ Suy luận: {1/fps:.4f}s, FPS={fps:.2f}")

# ====== Giải mã YOLOv8 output ======
preds = np.squeeze(preds).T  # (N, 8)
boxes = preds[:, :4]
obj_conf = preds[:, 4]
cls_conf = preds[:, 5:]
cls_ids = np.argmax(cls_conf, axis=1)
cls_scores = cls_conf[np.arange(len(cls_conf)), cls_ids]
conf = obj_conf * cls_scores

# Lọc theo ngưỡng tin cậy
mask = conf > conf_thresh
cls_ids = cls_ids[mask]
conf = conf[mask]

# ====== In nhãn ======
if len(cls_ids) == 0:
    print("❌ Không phát hiện vật thể nào.")
else:
    print(f"✅ Phát hiện {len(cls_ids)} vật thể:")
    for i, cls in enumerate(cls_ids):
        print(f" - {class_names[int(cls)]} ({conf[i]:.2f})")
