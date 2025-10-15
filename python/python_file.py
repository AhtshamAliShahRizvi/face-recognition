import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
# Load known faces (assume images are in the same directory; adjust paths if needed)
try:
    ahtsham_image = face_recognition.load_image_file("images\ahtsham_image.jpg")
    ahtsham_encoding = face_recognition.face_encodings(ahtsham_image)[0]

    ahtram_image = face_recognition.load_image_file("images\ahtram_image.jpg")
    ahtram_encoding = face_recognition.face_encodings(ahtram_image)[0]

    nadeem_image = face_recognition.load_image_file("images\nadeem_image.jpg")
    nadeem_encoding = face_recognition.face_encodings(nadeem_image)[0]

    wahid_image = face_recognition.load_image_file("images\wahid_image.jpg")
    wahid_encoding = face_recognition.face_encodings(wahid_image)[0]
except IndexError:
    print("Error: No faces found in known images.")
    exit()
except FileNotFoundError:
    print("Error: Image files not found. Check paths.")
    exit()

known_face_encodings = [ahtsham_encoding, ahtram_encoding,nadeem_encoding,wahid_encoding]
known_face_names = ["ahtsham", "ahtram","nadeem","wahid"]