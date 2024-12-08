import cv2

video_path = '/Users/nilufarkurbonova/Desktop/same_instance.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Resize for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Detect people
    found, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Draw bounding boxes
    for (x, y, w, h) in found:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('People Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
