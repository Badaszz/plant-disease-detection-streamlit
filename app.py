import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

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
        # Read the image file using OpenCV
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', channels='BGR')
        st.write("Classifying...")
        
        # Preprocess the image for the model
        image = cv2.resize(image, (256, 256))  # Resize to the input size of the model
        
        #Convert the image to a 4D tensor
        image.shape = (1, 256, 256, 3)  
        
        #Make prediction
        prediction = model.predict(image)
        result = class_names[np.argmax(prediction)]
        
        st.title(f"This is the {result.split('-')[0]} plant and it is affected by {result.split('-')[1]} disease.")
        st.balloons()    