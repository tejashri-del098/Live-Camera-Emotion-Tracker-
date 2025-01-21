# People and Emotion Tracking System
This project implements a real-time system for tracking people and detecting their emotions using a webcam feed. The program uses OpenCV for face detection and a simplified emotion detection mechanism to display the results on the video feed.

## Features
Detects faces in real-time using OpenCV's Haar cascade classifier.
Displays a rectangle around detected faces.
Assigns a random emotion to each face (as a placeholder for advanced emotion recognition models).
Tracks and displays the number of people detected in the frame.
## Prerequisites
Make sure the following dependencies are installed on your system:

- Python 3.6 or later
- OpenCV (cv2)
- NumPy

You can install the required Python libraries using pip:
```
pip install opencv-python-headless numpy
```

## How to Run
Ensure you have a webcam connected to your system.
Save the script as main.py.
Open a terminal or command prompt and navigate to the directory containing main.py.
Run the script:

```
python main.py
```
A window will appear displaying the video feed. Detected faces will have rectangles drawn around them, and their associated emotions will be displayed.
Controls
Press q to quit the application.


## Project Structure
### EmotionTracker class:
Manages face detection, emotion mapping, and tracking the number of people in the frame.
main() function: Initializes the webcam, processes each frame using the EmotionTracker, and displays the annotated video feed.
Limitations
The emotion detection mechanism is a placeholder and randomly assigns emotions. For real-world applications, integrate a proper emotion recognition model.
Requires a reasonably well-lit environment for accurate face detection.
Future Enhancements
Integrate a deep learning model for accurate emotion recognition (e.g., using TensorFlow or PyTorch).
Add support for saving detected data (e.g., people count over time) to a file or database.
Extend the system to handle multiple camera feeds.
