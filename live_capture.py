cap = cv2.VideoCapture(0)  # Webcam (or pass video path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)  # Inference
    results[0].show()       # Show first (and only) result
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()