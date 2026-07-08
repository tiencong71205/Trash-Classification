import cv2

def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# Th? 2 pipeline khác nhau: CSI ho?c USB
PIPELINES = [
    gstreamer_pipeline(sensor_id=0),           # CSI camera (Jetson)
    "v4l2src device=/dev/video0 ! videoconvert ! appsink"  # USB camera fallback
]

cap = None
for pipeline in PIPELINES:
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"? Camera connected with pipeline:\n{pipeline}")
        break

if not cap or not cap.isOpened():
    print("? Failed to open any camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("?? Frame not captured.")
            continue

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

