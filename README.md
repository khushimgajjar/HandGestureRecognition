# Hand Gesture Recognition

This Python script utilizes the OpenCV and Mediapipe libraries to perform real-time hand gesture recognition using your webcam. The program captures video frames, processes them to detect hand landmarks, and identifies gestures such as "Okay," "Peace," "Thumbs Up," "Stop," and "Thumbs Down."

## Setup

Make sure you have the required libraries installed:

```bash
pip install opencv-python mediapipe
```

## How to Use

1. Run the script.
2. Your webcam will activate, and the program will start detecting and recognizing hand gestures.
3. Press 'q' to exit the program.

## Supported Gestures

- **Okay**: Detects if the thumb and index finger form an 'Okay' gesture.
- **Peace**: Recognizes the 'Peace' sign by checking the position of the index and middle fingers.
- **Thumbs Up**: Identifies a 'Thumbs Up' gesture by comparing the positions of the thumb and index finger.
- **Stop**: Recognizes a 'Stop' gesture based on the angles between specific landmarks on the hand.
- **Thumbs Down**: Detects a 'Thumbs Down' gesture by analyzing the relative positions of the thumb and index finger.

Feel free to explore and modify the code to add more gestures or customize the recognition logic based on your requirements.
