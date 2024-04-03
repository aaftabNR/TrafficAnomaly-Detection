import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report

def load_data(folder_path, target_size=(224, 224)):
    data = []
    labels = []
    classes = os.listdir(folder_path)

    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        for frame_name in os.listdir(class_path):
            frame_path = os.path.join(class_path, frame_name)
            try:
                frame = cv2.imread(frame_path)  # Assuming frames are stored as images
                if frame is not None:  # Check if the image was successfully read
                    frame = cv2.resize(frame, target_size)  # Resize image
                    data.append(frame)
                    labels.append(class_name)
                else:
                    print(f"Error reading image {frame_path}: Image is None")
            except Exception as e:
                print(f"Error reading or resizing image {frame_path}: {str(e)}")

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def extract_features(data):
    features = model.predict(data)
    return features

# Load Preprocessed Data (frames) directly from "Dataset" folder
folder_path = r"D:\PBL2 project\Dataset\Optical_Flow_Frames"
data, labels = load_data(folder_path)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load Pre-trained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

# Extract Features
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Train Anomaly Detection Model (One-Class SVM)
model_anomaly = OneClassSVM()
model_anomaly.fit(X_train_features)

# Sample a fraction of the training data (e.g., 50%)
sample_fraction = 0.5
sample_indices = np.random.choice(len(X_train_features), int(sample_fraction * len(X_train_features)), replace=False)
X_train_features_sampled = X_train_features[sample_indices]

# Train the One-Class SVM model on the sampled data
model_anomaly.fit(X_train_features_sampled)

# Test Anomaly Detection Model
predictions = model_anomaly.predict(X_test_features)

# Evaluate the Model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
