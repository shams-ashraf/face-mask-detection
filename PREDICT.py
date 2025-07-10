import cv2
import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from preprocess import compute_hog,compute_glcm_features

def resize_pad(img, size=(64,64)):
    h,w = img.shape[:2]
    tw,th = size
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.zeros((th, tw), dtype=np.uint8)
    x_off, y_off = (tw-nw)//2, (th-nh)//2
    padded[y_off:y_off+nh, x_off:x_off+nw] = resized
    return padded

pca = joblib.load("files/pca_model.pkl")
svm = joblib.load("files/svm_face_mask_model.pkl")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Face Mask Detection GUI")
lbl = tk.Label(root)
lbl.pack()

def update():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        if x<0 or y<0 or x+w>gray.shape[1] or y+h>gray.shape[0]:
            continue
        face = resize_pad(gray[y:y+h, x:x+w])
        face = cv2.normalize(face, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)

        feats = np.concatenate((compute_hog(face), compute_glcm_features(face))).reshape(1,-1)
        feats_pca = pca.transform(feats)
        pred = svm.predict(feats_pca)[0]

        text = "With Mask" if pred==1 else "Without Mask"
        color = (0,255,0) if pred==1 else (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    lbl.imgtk = img
    lbl.configure(image=img)
    root.after(10, update)

def quit_app():
    cap.release()
    root.destroy()

btn_quit = tk.Button(root, text="Quit", command=quit_app)
btn_quit.pack()
root.protocol("WM_DELETE_WINDOW", quit_app)
update()
root.mainloop()
