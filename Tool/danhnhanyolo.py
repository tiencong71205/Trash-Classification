from ultralytics import YOLO
import os
import cv2

model = YOLO("yolo11x.pt")

img_dir = "glass"
label_dir = "glass"
os.makedirs(label_dir, exist_ok=True)

CONF_LEVELS = [0.9,0.7,0.5,0.3, 0.2, 0.1]

img_ext = [".jpg", ".png", ".jpeg"]
images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_ext]

total_new_labels = 0

for img_name in images:
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    if os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Không đọc được ảnh: {img_name}")
        continue
    h, w = img.shape[:2]

    detections = []
    for conf in CONF_LEVELS:
        results = model(img_path, conf=conf)
        boxes = results[0].boxes.data.tolist()

        if len(boxes) > 0:
            detections = boxes
            print(f"🔎 {img_name}: phát hiện được {len(boxes)} box ở conf={conf}")
            break

    if not detections:
        print(f"❌ Không phát hiện được gì cho {img_name}")
        continue

    with open(label_path, "w") as f:
        for r in detections:
            x1, y1, x2, y2, score, cls = r

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"0 {x_centerd:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ Đã tạo nhãn cho: {img_name}")
    total_new_labels += 1

print(f"\n🎯 Hoàn thành! Đã tạo nhãn mới cho {total_new_labels} ảnh chưa có nhãn.")