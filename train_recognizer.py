import cv2
import os
import numpy as np
import pickle

known_faces_dir = "known_faces"
faces = []
labels = []
label_dict = {}
current_id = 0
IMG_SIZE = (200, 200)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def augment(img):
    augmented = [img, cv2.flip(img, 1)]
    rows, cols = img.shape
    for angle in [10, -10]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        augmented.append(cv2.warpAffine(img, M, (cols, rows)))
    return augmented

for person_name in os.listdir(known_faces_dir):
    person_folder = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_file in os.listdir(person_folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces_detected:
            face_region = img[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, IMG_SIZE)
            face_region = cv2.equalizeHist(face_region)
            for aug_img in augment(face_region):
                faces.append(aug_img)
                labels.append(current_id)

    label_dict[current_id] = person_name
    current_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(faces, labels)
recognizer.save("lbph_model.yml")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_dict, f)

print("Training complete!")
