import cv2
import numpy as np
from datetime import datetime


class EmotionTracker:
    def __init__(self):
        # Initialize the face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Simple emotion mapping based on basic features
        self.emotions = ['neutral', 'happy', 'sad', 'angry']

        # Counter for people
        self.person_count = 0

        # Dictionary to store person tracks
        self.tracks = {}

    def detect_emotion(self, face_roi):
        # This is a simplified emotion detection
        # In a real implementation, you would use a proper emotion recognition model
        # For demo purposes, we'll return a random emotion
        return np.random.choice(self.emotions)

    def process_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Update person count
        self.person_count = len(faces)

        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get face ROI
            face_roi = frame[y:y + h, x:x + w]

            # Detect emotion
            emotion = self.detect_emotion(face_roi)

            # Draw emotion text
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Draw person count
        cv2.putText(frame, f'People count: {self.person_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame


def main():
    print("Starting People and Emotion Tracking System...")

    # Initialize video capture (0 for webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    # Initialize tracker
    tracker = EmotionTracker()

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame")
            break

        # Process frame
        processed_frame = tracker.process_frame(frame)

        # Display frame
        cv2.imshow('People and Emotion Tracking', processed_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

