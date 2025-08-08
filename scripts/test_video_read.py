video_path = r"C:/Users/bibek/deepfake-project/data/raw_videos/real/v1.mp4"

import cv2

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Could not open video.")
else:
    print("[SUCCESS] Video opened successfully!")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
