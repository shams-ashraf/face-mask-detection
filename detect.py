import cv2
import joblib
import numpy as np
import os
from preprocess import compute_hog, compute_glcm_features

def extract_features(img):
    return np.concatenate((compute_hog(img), compute_glcm_features(img)))

os.makedirs('result', exist_ok=True)
model = joblib.load("files/svm_face_mask_model.pkl")
pca = joblib.load("files/pca_model.pkl")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for f in os.listdir('images'):
    if not f.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    img = cv2.imread(os.path.join('images', f))
    if img is None:
        print(f"Could not read image: {f}")
        continue

    h, w = img.shape[:2]
    if min(h,w) < 200:
        scale = 200 / min(h,w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces) == 0:
        print(f"{f}: No faces detected")
        continue

    for (x,y,w,h) in faces:
        x, y, w, h = max(0,x), max(0,y), max(1,w), max(1,h)
        w = min(w, gray.shape[1]-x)
        h = min(h, gray.shape[0]-y)
        if w <= 0 or h <= 0:
            continue

        face = cv2.resize(gray[y:y+h, x:x+w], (64,64))
        face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        features = extract_features(face).reshape(1, -1)
        reduced = pca.transform(features)
        pred = model.predict(reduced)[0]

        label = "Mask" if pred == 1 else "No Mask"
        color = (0,255,0) if pred == 1 else (0,0,255)

        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(f"{f}: {label}")

    cv2.imwrite(os.path.join('result', f), img)
