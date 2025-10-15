import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
# Load known faces (assume images are in the same directory; adjust paths if needed)
try:
    ahtsham_image = face_recognition.load_image_file("images/ahtsham_image.jpg")
    ahtsham_encoding = face_recognition.face_encodings(ahtsham_image)[0]

    ahtram_image = face_recognition.load_image_file("images/ahtram_image.jpg")
    ahtram_encoding = face_recognition.face_encodings(ahtram_image)[0]

    nadeem_image = face_recognition.load_image_file("images/nadeem_image.jpg")
    nadeem_encoding = face_recognition.face_encodings(nadeem_image)[0]

    wahid_image = face_recognition.load_image_file("images/wahid_image.jpg")
    wahid_encoding = face_recognition.face_encodings(wahid_image)[0]

except IndexError:
    print("Error: No faces found in known images.")
    exit()
except FileNotFoundError:
    print("Error: Image files not found. Check paths.")
    exit()

known_face_encodings = [ahtsham_encoding, ahtram_encoding,nadeem_encoding,wahid_encoding]
known_face_names = ["ahtsham", "ahtram","nadeem","wahid"]

# List of expected students
students = known_face_names.copy()

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get current date for CSV
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(f"{current_date}.csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])  # Header row

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

# Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Mark attendance if not already done
                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    f.flush()  # Ensure written to file immediately

        # Draw annotations on the original frame (scale locations back up)
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()