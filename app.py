import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#CIFAR-10 class labels
class_names=['airplane', 
             'automobile', 
             'bird', 
             'cat', 
             'deer', 
             'dog', 
             'frog', 
             'horse', 
             'ship', 
             'truck'
            ]
st.title("CIFAR-10 Image Classification App")
st.write("Upload an image and the trained CNN model will classify it.")

#Load model
def load_model():
    model = tf.keras.models.load_model("regularized_cnn_cifar10.keras")
    return model

model = load_model()

#upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Uploaded Image", use_column_width=True)
  image = image.resize((32,32))
  img_array = np.array(image) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  prediction = model.predict(img_array)
  predicted_class = class_names[np.argmax(prediction)]
  confidence = np.max(prediction)
  st.success(f"Prediction: {predicted_class}")
  st.write(f"Confidence: {confidence*100:.2f}%")
  






