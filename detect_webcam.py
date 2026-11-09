import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Choose ONE camera source by uncommenting its line and commenting out the other.

# Local webcam (0 is usually the built-in/default camera; try 1 for another camera)
# cap = cv2.VideoCapture(0)

# Phone camera over Wi-Fi (Android: install/open "IP Webcam" and start its server;
# use the address shown by the app, e.g. http://192.168.1.50:8080/video).
# Keep the phone and computer on the same Wi-Fi network.
cap = cv2.VideoCapture("http://192.168.254.105:8080/video", cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Could not open the selected camera. Check the camera index or phone stream URL.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Run inference on current video frame
    results = model.predict(source=frame, conf=0.4, verbose=False)

    # Draw bounding boxes onto the frame
    annotated_frame = results[0].plot()

    # Show live detection window
    cv2.imshow("OBVIATE - Recyclable Detector", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()