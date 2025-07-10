import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import joblib
from skimage.feature import graycomatrix, graycoprops

def prep(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None or np.std(img) < 1:
        return None
    img = cv2.resize(img, (64, 64))
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compute_gradients(image):
    gx = image[:, 1:] - image[:, :-1]
    gy = image[1:, :] - image[:-1, :]
    gx = np.pad(gx, ((0, 0), (0,1)), mode='constant')
    gy = np.pad(gy, ((0,1), (0,0)), mode='constant')
    magnitude = cv2.magnitude(gx, gy)
    orientation = cv2.phase(gx, gy, angleInDegrees=True) % 180
    return magnitude, orientation

def compute_histograms(mag, ori, cell_size=8, bins=9):
    h, w = mag.shape
    hist = np.zeros((h // cell_size, w // cell_size, bins))
    bin_width = 180 / bins
    for i in range(0, h, cell_size):
        for j in range(0, w, cell_size):
            for u in range(cell_size):
                for v in range(cell_size):
                    y = i + u
                    x = j + v
                    if y >= h or x >= w: 
                        continue
                    angle = ori[y, x]
                    magnitude_val = mag[y, x]
                    bin_idx = angle / bin_width
                    lb = int(np.floor(bin_idx)) % bins
                    ub = (lb + 1) % bins
                    ratio = bin_idx - lb
                    hist[i // cell_size, j // cell_size, lb] += magnitude_val * (1 - ratio)
                    hist[i // cell_size, j // cell_size, ub] += magnitude_val * ratio
    return hist

def normalize_blocks(hist, block_size=2):
    h, w, b = hist.shape
    hog_vector = []
    for i in range(h - block_size + 1):
        for j in range(w - block_size + 1):
            block = hist[i:i+block_size, j:j+block_size, :].ravel()
            norm = np.sqrt(np.sum(block**2) + 1e-5)
            hog_vector.extend(block / norm)
    return np.array(hog_vector)

def compute_hog(image):
    mag, ori = compute_gradients(image.astype(np.float32))
    hist = compute_histograms(mag, ori)
    return normalize_blocks(hist)

def compute_glcm_features(image):
    image = cv2.resize(image, (64, 64))
    image = (image // 16).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0]
    energy = graycoprops(glcm, 'energy')[0]
    correlation = graycoprops(glcm, 'correlation')[0]
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
    return features

def apply_pca(features, n_components=100):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    print(f"PCA: Reduced from {features.shape[1]} → {reduced.shape[1]}")
    return reduced, pca

def extract_and_save_features_with_pca(data_folder="dataset", pca_components=100):
    X, y = [], []
    label_map = {"without_mask": 0, "with_mask": 1}
    for label_name, label in label_map.items():
        folder = os.path.join(data_folder, label_name)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        print(f"{label_name}: {len(images)} images")
        for idx, fname in enumerate(images):
            img_path = os.path.join(folder, fname)
            print(f"Processing {fname}")
            img = prep(img_path)
            if img is None: 
                continue
            hog = compute_hog(img)
            glcm = compute_glcm_features(img)
            features = np.concatenate((hog, glcm))
            X.append(features)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    X_reduced, pca_model = apply_pca(X, pca_components)
    os.makedirs("files", exist_ok=True)
    np.save("files/features_pca.npy", X_reduced)
    np.save("files/labels.npy", y)
    joblib.dump(pca_model, "files/pca_model.pkl")
    print("Features, labels, and PCA model saved.")

if __name__ == "__main__":
    extract_and_save_features_with_pca()
