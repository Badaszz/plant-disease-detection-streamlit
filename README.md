# 🌿 Plant Disease Detection App

This project uses a Convolutional Neural Network (CNN) to detect plant diseases from leaf images. The model is deployed using Streamlit for a user-friendly web interface.

## 🚀 Features

- Upload a plant leaf image
- Get instant prediction of disease type
- Trained CNN model with ~90% accuracy on small dataset

## 🧠 Model Info

- Model: Convolutional Neural Network (Keras)
- Classes: `Corn-Common_rust`, `Potato-Early_blight`, `Tomato-Bacterial_spot`
- Accuracy: ~90%

## 🖥️ Web App

Built with Streamlit. Upload a leaf image and get a prediction instantly.

## 🧪 How to Use
pip install -r requirements.txt
streamlit run app.py

## Model Download

Download the trained model here 👉 [Google Drive Link]  
After downloading, place the `plant_model.h5` file in the same folder as `app.py`.
