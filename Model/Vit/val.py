import cv2
import torch
from PIL import Image
from torchvision import transforms


def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def load_pt_model(model_path="trash_15k_3_cls_model.pt", use_cuda=False):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.eval()

    print(f"[INFO] Loaded regular PyTorch model from {model_path} on {device}")
    return model, device

# ==== FRAME PROCESSING ====
def classify_frame(frame, model, device, label_map):
    # Chu?n hóa ?nh (gi?ng khi train)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        conf = probs[0, pred_id].item()

    label = label_map.get(pred_id, "Unknown")
    return label, conf


# ==== MAIN CAMERA LOOP ====
def run_camera(model, device, label_map, conf_threshold=0.5):
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Camera started. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame not captured.")
                continue

            label, conf = classify_frame(frame, model, device, label_map)

            color = (0, 255, 0) if conf >= conf_threshold else (0, 0, 255)
            cv2.putText(frame, f"{label} {conf:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Camera Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==== RUN ====
if __name__ == "__main__":
    LABEL_MAP = {
        0: "Glass",
        1: "Paper",
        2: "Plastic"
    }

    model, device = load_pt_model("trash_15k_3_cls_model.pt", use_cuda=False)
    run_camera(model, device, LABEL_MAP, conf_threshold=0.5)

