import cv2

cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
