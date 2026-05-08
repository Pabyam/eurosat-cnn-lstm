import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import CNNLSTM

# Class names in the same order as EuroSAT
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Load the model (cached to avoid reloading every time)
@st.cache_resource
def load_model():
    model = CNNLSTM(input_channels=3, num_classes=10)
    model.load_state_dict(torch.load('best_model_rgb.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """Convert uploaded image to 3×64×64 tensor normalized to [0,1]."""
    image = image.convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.tensor(img_array, dtype=torch.float32)
    return tensor.unsqueeze(0)  # add batch dimension

st.set_page_config(page_title="EuroSAT Land-Use Classifier", page_icon="🌍")
st.title("🌍 EuroSAT Land-Use Classification")
st.write("Upload a 64×64 RGB satellite image patch to classify the land‑use type.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Classifying...'):
        tensor = preprocess_image(image)
        model = load_model()
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            class_name = CLASS_NAMES[predicted.item()]
    
    st.success(f"**Prediction:** {class_name}")
    st.balloons()
