import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# Define a function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the vertical distances between the eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the horizontal distance between the eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face detector and landmark predictor models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define constants for the EAR threshold and number of consecutive frames
# the EAR must be below the threshold to indicate a blink
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 3

# Initialize the frame counter and blink counter
frame_counter = 0
blink_counter = 0

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over the detected faces
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Get the left and right eye landmarks
        left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                             (landmarks.part(37).x, landmarks.part(37).y),
                             (landmarks.part(38).x, landmarks.part(38).y),
                             (landmarks.part(39).x, landmarks.part(39).y),
                             (landmarks.part(40).x, landmarks.part(40).y),
                             (landmarks.part(41).x, landmarks.part(41).y)], dtype=np.int32)

        right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                              (landmarks.part(43).x, landmarks.part(43).y),
                              (landmarks.part(44).x, landmarks.part(44).y),
                              (landmarks.part(45).x, landmarks.part(45).y),
                              (landmarks.part(46).x, landmarks.part(46).y),
                              (landmarks.part(47).x, landmarks.part(47).y)], dtype=np.int32)

        # Calculate the eye aspect ratio (EAR) for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Compute the average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Draw the eye landmarks on the frame
        cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)

        # Check if the EAR is below the threshold
        if ear < EAR_THRESHOLD: 
            # Increment the frame counter
            frame_counter += 1

            # If the EAR is below the threshold for the required number of consecutive frames,
            # increment the blink counter and reset the frame counter
            if frame_counter >= CONSECUTIVE_FRAMES and ear < EAR_THRESHOLD:
                blink_counter += 1
                frame_counter = 0

        # Display the frame with the eye landmarks and the blink counter
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

