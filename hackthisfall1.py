import subprocess
import sys

# Install opencv-python-headless if not already installed
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python-headless'])



import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# Configure your Gemini API key
genai.configure(api_key='AIzaSyC0FlgJ93BIdfBmhttXU591pg8wddnUg8M')  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Custom CSS for a water-inspired theme
st.markdown("""
    <style>
    /* Theme colors for a water-inspired look */
    body {
        background-color: #E6F7FF;
        font-family: 'Open Sans', sans-serif;
    }

    /* Header styling */
    .header {
        background: #1A73E8;
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }

    /* Banner styling for the landing page */
    .banner {
        background: linear-gradient(135deg, #74ebd5, #acb6e5);
        padding: 80px;
        text-align: center;
        color: white;
        border-radius: 10px;
        margin: 30px 0;
    }

    /* Adjusted Get Started and Analyze Another Image button container for center alignment */
    .center-button-container {
        display: flex;
        justify-content: center;
        margin-top: 30px;
    }
    .stButton>button {
        background-color: #1A73E8;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0b57d0;
        color: white;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 15px;
        font-size: 16px;
        color: #083D77;
        margin-top: 20px;
        background: #B3E5FC;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Function for the landing page
def landing_page():
    st.markdown('<div class="header">Water Purity Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="banner"><h2>Welcome to the Water Purity Analyzer</h2><p>Analyze the purity of water samples using AI technology.</p></div>', unsafe_allow_html=True)
    
    st.write("### Discover the Quality of Your Water")
    st.write("Upload an image of your water sample to receive an analysis of potential impurities. Our AI-powered tool provides insights into water purity to help ensure your water meets quality standards.")

    # Right-aligned "Get Started" Button
    st.markdown('<div class="center-button-container">', unsafe_allow_html=True)
    if st.button("Get Started"):
        st.session_state.page = "main"
    st.markdown('</div>', unsafe_allow_html=True)

# Function to compare images using color moments
def compare_images(img1_path, img2_file):
    img1 = cv2.imread(img1_path)
    img2 = Image.open(img2_file)
    img2 = np.array(img2)

    if img2.shape[-1] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)
    elif img2.shape[-1] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    moments1 = cv2.moments(gray1)
    hu_moments1 = cv2.HuMoments(moments1)[0]
    moments2 = cv2.moments(gray2)
    hu_moments2 = cv2.HuMoments(moments2)[0]
    color_moments_distance = np.linalg.norm(hu_moments1 - hu_moments2)

    return color_moments_distance

# Function to find the closest match for impurity level
def determine_impurity_level(distances):
    labels = ["0%", "25%", "50%", "75%", "100%"]
    min_distance_label = labels[np.argmin(distances)]
    return min_distance_label

# Generate comments from Gemini based on impurity level
def generate_gemini_comments(impurity_level):
    prompt = (
        f"Gemini Pro Water Purity Analysis: The predicted water impurity level is {impurity_level}. "
        "Please provide comments on what a water impurity level of this percentage implies about the water quality."
    )
    try:
        response = model.generate_content(prompt)
        return response.text if response and response.text else "Our model did not provide any comments on this level of water impurity."
    except Exception as e:
        return f"An error occurred while generating comments from our model: {str(e)}"

# Main analysis page function
def analysis_page():
    st.markdown('<div class="header">Water Purity Analysis</div>', unsafe_allow_html=True)
    uploaded_test_image = st.file_uploader("Upload a test image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_test_image:
        st.image(uploaded_test_image, caption="Uploaded Test Image", use_container_width=True)
        
        label_dir = "C:/_mksp/Xylem_Hackathon/label_images"
        label_images = ["0.jpg", "25.jpg", "50.jpg", "75.jpg", "100.jpg"]
        
        distances = []
        for label in label_images:
            label_path = f"{label_dir}/{label}"
            try:
                distance = compare_images(label_path, uploaded_test_image)
                distances.append(distance)
            except FileNotFoundError:
                st.error(f"Labeled image {label} not found at path {label_path}.")
                return

        impurity_level = determine_impurity_level(distances)
        st.write(f"**Predicted Water Impurity Level:** {impurity_level}")

        gemini_comments = generate_gemini_comments(impurity_level)
        st.write("### Our Model's Analysis")
        st.write(gemini_comments)

        # Centered "Analyze Another Image" button
        st.markdown('<div class="center-button-container">', unsafe_allow_html=True)
        if st.button("🔄 Analyze Another Image"):
            st.session_state.page = "landing"
        st.markdown('</div>', unsafe_allow_html=True)

# Navigation with session state
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    landing_page()
else:
    analysis_page()

# Footer with custom text
st.markdown('<div class="footer">Designed by Srepadmashiny K & Sree Ranjane M K for Hack This Fall 2024 Virtual Hackathon</div>', unsafe_allow_html=True)
