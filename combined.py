import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the combined model
combined_model = load_model('combined_model.h5')

# Define the path to the image
img_path = 'D:/PBL2_project/Dataset/Accident Detection From CCTV Footage/data/val/Accident/test7_32.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

# Use the combined model to make predictions
prediction = combined_model.predict([img_array, img_array])

if prediction.shape[1] == 2:
    medical_prediction = prediction[0][0]  # Probability of fall (class 0)
    traffic_prediction = prediction[0][1]  # Probability of accident (class 1)
else:
    medical_prediction = prediction[0][0]  # Probability of fall (class 0)
    traffic_prediction = 1 - medical_prediction  # Probability of accident (class 1)


# Define threshold for classification
threshold = 0.5

# Determine the class based on the threshold
medical_class = 'Fall' if medical_prediction > threshold else 'No Fall'
traffic_class = 'Accident' if traffic_prediction > threshold else 'No Accident'

# Load ground truth labels (replace these with your actual ground truth labels)
# Example: ground_truth_medical = 'Fall'
#          ground_truth_traffic = 'Accident'
ground_truth_medical = 'Accident'
ground_truth_traffic = 'Accident'

# Calculate accuracy
medical_accuracy = 1 if medical_class == ground_truth_medical else 0
traffic_accuracy = 1 if traffic_class == ground_truth_traffic else 0
overall_accuracy = (medical_accuracy + traffic_accuracy) / 2

# Plot the image and print the predicted classes and accuracy
plt.imshow(img)
plt.title(f'Medical Class: {medical_class}, Traffic Class: {traffic_class}, Overall Accuracy: {overall_accuracy}')
plt.axis('off')
plt.show()
