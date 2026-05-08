# 🌍 EuroSAT Land-Use Classification with CNN+LSTM

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eurosat-cnn-lstm.streamlit.app/)

This project classifies satellite image patches into **10 land‑use categories** using a hybrid **CNN + LSTM** model trained on the EuroSAT RGB dataset.

- **CNN** extracts spatial features (3 convolutional + pooling blocks)
- **LSTM** processes the sequence of CNN feature channels to capture spectral dependencies
- Achieves **~93% accuracy** on the test set

## Live Demo

👉 [Click here to try the app](https://eurosat-cnn-lstm.streamlit.app/)  


## How to Run Locally

1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/eurosat-cnn-lstm.git
   cd eurosat-cnn-lstm
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit App
   ```bash
   streamlit run app.py
4. Open your browser at http://localhost:8501 and upload an RGB image patch (any size – it will be resized to 64×64).
