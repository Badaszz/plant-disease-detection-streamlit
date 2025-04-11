import numpy as np
import streamlit as st
#import cv2
from keras.models import load_model
from PIL import Image

import os
print(os.getcwd())

# Get the path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'plant_model.h5')

# Load the pre-trained modela
model = load_model(file_path)

# Define the class names
class_names = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#setting the title of the app
st.title("Plant Disease Detection App")
st.markdown("Upload an image of a plant leaf to detect the disease.")
st.write("This app uses a pre-trained CNN model to classify plant diseases.")

#Uploading the plant image
plant_image = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])
st.write("Please upload an image of a plant leaf to get started.")
submit = st.button("Submit")

#when the submit button is clicked
if submit:
    if plant_image is not None:
        # Read the image file using PIL
        image = Image.open(plant_image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        # Preprocess the image for the model
        image = image.resize((256, 256))  # Resize to the input size of the model

        # Convert the image to a NumPy array (RGB format)
        image = np.array(image)

        # Convert to a 4D tensor (batch size, height, width, channels)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Normalize the image if required by the model
        image = image / 255.0  # If your model was trained with normalized images

        # Make prediction
        prediction = model.predict(image)
        result = class_names[np.argmax(prediction)]

        # Display the result
        st.title(f"This is the {result.split('-')[0]} plant and it is affected by {result.split('-')[1]} disease.")
        st.balloons()
