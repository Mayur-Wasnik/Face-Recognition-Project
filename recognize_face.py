import face_recognition
import cv2
import pickle
import numpy as np

# --- 1. Load the data ---
ENCODINGS_FILE = 'face_encodings.pkl'
TOLERANCE = 0.6  # lower = stricter match, default in compare_faces is 0.6

# Load the data dictionary from the file
print("Loading known face encodings...")
with open(ENCODINGS_FILE, 'rb') as f:
    data = pickle.load(f)

# Unpack the data into separate lists
known_face_encodings = data['encodings']
known_face_names = data['names']

print(f"Successfully loaded {len(known_face_names)} known faces.")

# Initialize the video capture object

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Continue in recognize_face.py

# The Main Program Loop
while True:
    # 1. Read the frame from the camera
    ret, frame = video_capture.read()

    # If the frame was not read correctly, break the loop
    if not ret:
        break
    
    # --- Processing for face recognition goes here ---
    # 2. Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 3. Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # 4. Loop through each face found in the frame to see if it's someone we know
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        # Compute distances to all known faces and pick the best match
        if len(known_face_encodings) > 0:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_index = np.argmin(distances)
            best_distance = float(distances[best_index])
            if best_distance <= TOLERANCE:
                name = known_face_names[best_index]

        # 5. Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 6. Draw a label with a name (and distance) below the face
        label_text = name
        if name != "Unknown":
            label_text = f"{name} ({best_distance:.2f})"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label_text, (left + 6, bottom - 8), font, 0.6, (255, 255, 255), 1)

        # --- FINAL DISPLAY AND EXIT CONDITION ---
    # Display the resulting image
    cv2.imshow('Face Recognition System', frame)

    # Wait for 'q' key press to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all OpenCV windows
# Continue in recognize_face.py (OUTSIDE the 'while True:' loop)

# Release the webcam capture object
video_capture.release()

# Close all OpenCV display windows
cv2.destroyAllWindows()

print("\nProgram terminated and resources released.")
