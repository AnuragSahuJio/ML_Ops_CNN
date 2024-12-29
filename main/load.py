import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to the trained model
model_path = r"F:\ml engineering projects\PlantVillage\saved_model.h5"  # Update this path to your saved model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Path to the folder containing new images for prediction
prediction_dir = r"F:\ml engineering projects\PlantVillage\test_images"  # Update this to your test images directory

# Parameters for preprocessing
image_size = (128, 128)  # Ensure this matches the input size of your trained model
class_names = ['Healthy', 'Diseased']  # Replace with the actual class names you used during training

def predict_image(image_path, model):
    """
    Preprocess an image and make a prediction.
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]
    
    return predicted_label, predictions[0]

# Loop through test images and make predictions
print("Making predictions on images in:", prediction_dir)
for filename in os.listdir(prediction_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image file formats
        image_path = os.path.join(prediction_dir, filename)
        label, probabilities = predict_image(image_path, model)
        print(f"Image: {filename}, Predicted Label: {label}, Probabilities: {probabilities}")