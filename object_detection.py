import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the current frame
    results = model(frame)

    # Plot results
    annotated_frame = results[0].plot()

    # Resize for smaller window
    im_resized = cv2.resize(annotated_frame, (600, 400))

    # Show frame
    cv2.imshow("Detections", im_resized)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
