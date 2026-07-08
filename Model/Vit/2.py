import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("? Không m? du?c camera.")
    exit()

print("? Camera m? thành công. Nh?n Q d? thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("?? Không d?c du?c frame.")
        continue

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

