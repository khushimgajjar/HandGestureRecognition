import cv2
import mediapipe as mp
import math

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Define a function to check if the hand is making the "Okay" gesture
def is_okay_gesture(hand_landmarks):
    if hand_landmarks is not None:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        little_tip = hand_landmarks.landmark[20]

        # Calculate distances or angles between landmarks to recognize gestures
        # Example: Check distance between thumb tip and index tip for 'okay' gesture
        distance_thumb_index = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        # Implement similar checks for other gestures
        gesture_name = None
        if distance_thumb_index < 0.05:
            gesture_name = "Okay"
        # Add conditions for recognizing other gestures here

        return gesture_name

    return None

# Define a function to check if the hand is making the "Peace" gesture
def is_peace_gesture(hand_landmarks):
    if hand_landmarks is not None:
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]

        # Check if the tip of the index finger is above the tip of the middle finger
        if index_tip.y < middle_tip.y:
            return "Peace"

    return None

# Define a function to check if the hand is making the "Thumbs Up" gesture
def is_thumbs_up_gesture(hand_landmarks):
    if hand_landmarks is not None:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        thumb_x, thumb_y = thumb_tip.x, thumb_tip.y
        index_x, index_y = index_tip.x, index_tip.y

        # Check if the thumb is raised above the index finger
        if thumb_y < index_y and abs(thumb_x - index_x) < 0.03:
            return "Thumbs Up"

    return None

# Define a function to check if the hand is making the "Stop" gesture
def is_stop_gesture(hand_landmarks):
    if hand_landmarks is not None:
        palm_landmarks = [
            hand_landmarks.landmark[0],  # Wrist
            hand_landmarks.landmark[5],  # Base of the pinky finger
            hand_landmarks.landmark[9],  # Base of the ring finger
            hand_landmarks.landmark[13], # Base of the middle finger
            hand_landmarks.landmark[17], # Base of the index finger
        ]

        # Calculate the angles between the landmarks
        angle1 = calculate_angle(palm_landmarks[1], palm_landmarks[2], palm_landmarks[3])
        angle2 = calculate_angle(palm_landmarks[2], palm_landmarks[3], palm_landmarks[4])

        # Check if the angles are within a certain range
        if 150 < angle1 < 180 and 150 < angle2 < 180:
            return "Stop"

    return None

# Define a function to check if the hand is making the "Thumbs Down" gesture
def is_thumbs_down_gesture(hand_landmarks):
    if hand_landmarks is not None:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        thumb_x, thumb_y = thumb_tip.x, thumb_tip.y
        index_x, index_y = index_tip.x, index_tip.y

        # Check if the thumb is below the index finger
        if thumb_y > index_y and abs(thumb_x - index_x) < 0.03:
            return "Thumbs Down"

    return None

# Define a function to calculate the angle between three landmarks
def calculate_angle(a, b, c):
    # Calculate vectors between landmarks
    v1 = [a.x - b.x, a.y - b.y]
    v2 = [c.x - b.x, c.y - b.y]

    # Calculate the dot product and magnitude of vectors
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Calculate the angle in degrees
    angle = math.degrees(math.acos(cosine_angle))

    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    detected_gesture = None

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize the "Okay" gesture
            okay_gesture = is_okay_gesture(landmarks)
            if okay_gesture:
                detected_gesture = okay_gesture

            # Recognize the "Peace" gesture
            peace_gesture = is_peace_gesture(landmarks)
            if peace_gesture:
                detected_gesture = peace_gesture

            # Recognize the "Thumbs Up" gesture
            thumbs_up_gesture = is_thumbs_up_gesture(landmarks)
            if thumbs_up_gesture:
                detected_gesture = thumbs_up_gesture

            # Recognize the "Stop" gesture
            stop_gesture = is_stop_gesture(landmarks)
            if stop_gesture:
                detected_gesture = stop_gesture

            # Recognize the "Thumbs Down" gesture
            thumbs_down_gesture = is_thumbs_down_gesture(landmarks)
            if thumbs_down_gesture:
                detected_gesture = thumbs_down_gesture

    if detected_gesture:
        cv2.putText(
            frame,
            f"Detected Gesture: {detected_gesture}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
