# Importing the necessary libraries
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Customizing Streamlit styles
st.set_page_config(page_title="MNIST Prediction App", page_icon="📊", layout="centered")

# custom styling
def add_custom_styling():
    st.markdown(
        """
        <style>
        /* General App Style */
        .stApp {
            background-color: #121212;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        /* Title and Headers */
        .stTitle {
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            font-size: 36px;
            margin-bottom: 10px;
        }
        .stHeader {
            color: #f0f0f0;
            text-align: center;
        }
        /* File Upload Section */
        .uploadedFile img {
            max-width: 90%;
            border: 4px solid #4CAF50;
            border-radius: 10px;
            margin: 10px auto;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.6);
        }
        /* Buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #3E8E41;
            transform: scale(1.05);
        }
        /* Prediction Box */
        .prediction-box {
            background-color: #4CAF50;
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.6);
        }
        /* Footer */
        footer {
            visibility: hidden;
        }
        .custom-footer {
            text-align: center;
            font-size: small;
            color: #f0f0f0;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_styling()

# Loading the model
model = tf.keras.models.load_model('MNIST.keras')

# Model labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Function to preprocess data
def process_image(image):

    image = image.convert('L')

    image = image.resize((28, 28))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0) 
    image = np.expand_dims(image, axis=-1) 
    return image

# Function to accept the image and make prediction
def predict(image):
    processed_image = process_image(image)
    predictions = model.predict(processed_image)
    return labels[np.argmax(predictions)]

# Streamlit app
st.title("MNIST Prediction App")
st.markdown(
    """
    <div style="text-align: left; margin-bottom: 30px;">
        <p>Welcome to the <strong>MNIST Prediction App</strong>!<br>
        Upload a handwritten digit image, and our AI-powered model will predict its value with precision.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Allowing users to upload files
uploaded_file = st.file_uploader(
    "📤 Upload an image (Supported formats: JPG, JPEG, PNG, BMP)", 
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(
        image, caption="Uploaded Image", use_container_width=True, 
        output_format="JPEG"
    )

    # Show spinner during prediction
    with st.spinner("🔍 Analyzing the image..."):
        prediction = predict(image)

    # Display the prediction
    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted Digit: {prediction}
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <div class="custom-footer">
        Created by <strong>Nicky</strong> | Powered by <strong>TensorFlow</strong> & <strong>Streamlit</strong>
    </div>
    """,
    unsafe_allow_html=True
)
