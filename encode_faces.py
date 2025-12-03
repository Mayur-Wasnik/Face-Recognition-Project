import face_recognition
import os
import pickle
import numpy as np

# Define the paths for input images and output encoding file
KNOWN_FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'face_encodings.pkl'
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Initialize lists to store our data
known_face_encodings = []
known_face_names = []

# List files and show what we'll try to process
files = os.listdir(KNOWN_FACES_DIR)
print(f"Found {len(files)} files in '{KNOWN_FACES_DIR}': {files}")

# Loop through every file in the known_faces directory
for filename in files:
    # Case-insensitive extension check
    if filename.lower().endswith(SUPPORTED_EXTENSIONS):

        # 1. Create the full path to the image
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        # 2. Extract the name from the filename (e.g., 'elon.jpg' -> 'elon')
        name = os.path.splitext(filename)[0]

        print(f"Processing '{filename}' (name='{name}')...")

        # 3. Load the image file
        try:
            image = face_recognition.load_image_file(image_path)
        except Exception as e:
            print(f"  Error loading {filename}: {e}. Skipping.")
            continue

        # 4. Get the face encodings for the face(s) in the image
        encodings = face_recognition.face_encodings(image)

        # 5. Check if a face was actually found
        if len(encodings) > 0:
            # If there are multiple faces in the same image, warn and take the first
            if len(encodings) > 1:
                print(f"  Warning: {filename} contains {len(encodings)} faces - using the first one.")
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            print(f"  Successfully encoded {name}.")
        else:
            print(f"  No face found in {filename}. Skipping.")
    else:
        print(f"Skipping '{filename}' (unsupported extension).")
# Save the encodings and names to a file using pickle
# Create the data dictionary to save
data = {
    "encodings": known_face_encodings,
    "names": known_face_names
}

# Save the encodings to a file
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump(data, f)

print(f"\n✅ All known faces encoded and saved to {ENCODINGS_FILE}")