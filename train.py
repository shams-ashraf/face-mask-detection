import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import time

def load_features_and_labels():
    X_data = np.load("files/features_pca.npy")    
    y_data = np.load("files/labels.npy")
    return train_test_split(X_data, y_data, test_size=0.2, random_state=42)

def train_and_compare_models():
    X_train, X_test, y_train, y_test = load_features_and_labels()

    models = {
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    accuracies = {}

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy

        train_time = end_time - start_time
        print(f"{name} Accuracy: {accuracy*100:.2f}% | Training Time: {train_time:.4f} seconds")

    joblib.dump(models["SVM"], "files/svm_face_mask_model.pkl")
    print("SVM model saved as svm_face_mask_model.pkl")

    return models, accuracies

if __name__ == "__main__":
    train_and_compare_models()
