import cv2
import numpy as np
import onnxruntime as ort
import time

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "refine_last_phase3_10epochs.onnx"
LABELS = ["paper", "plastic", "glass"]
CONF_THRESH = 0.25
NMS_THRESH = 0.45
INPUT_SIZE = 640

# -------------------------------
# Load model
# -------------------------------
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# GStreamer pipeline cho camera CSI (Jetson Nano)
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=2 ! "
    "video/x-raw, width=640, height=640, format=BGRx ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

print("✅ Camera đã khởi động, đang chạy nhận dạng...")

# -------------------------------
# Hàm xử lý dự đoán
# -------------------------------
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, 0)
    return img

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

# -------------------------------
# Loop camera
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không nhận được khung hình")
        continue

    h0, w0 = frame.shape[:2]
    img_in = preprocess(frame)

    # Dự đoán
    t1 = time.time()
    outputs = session.run(None, {input_name: img_in})[0]
    outputs = np.squeeze(outputs)
    outputs = outputs.transpose(1, 0)  # (8400, 84)

    boxes = outputs[:, :4]
    scores = outputs[:, 4]
    class_probs = outputs[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = scores * class_probs[np.arange(len(scores)), class_ids]

    # Lọc theo ngưỡng
    mask = confidences > CONF_THRESH
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
    if len(boxes) == 0:
        cv2.imshow("YOLOv8 ONNX", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Chuyển sang xyxy
    boxes = xywh2xyxy(boxes)

    # Tỷ lệ theo frame gốc
    scale_w, scale_h = w0 / INPUT_SIZE, h0 / INPUT_SIZE
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), CONF_THRESH, NMS_THRESH)

    # Vẽ bbox
    for i in indices:
        i = int(i)
        x1, y1, x2, y2 = map(int, boxes[i])
        cls_id = int(class_ids[i])
        conf = float(confidences[i])
        label = f"{LABELS[cls_id]} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fps = 1 / (time.time() - t1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 ONNX", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
