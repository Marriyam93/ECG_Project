import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load the model with caching
@st.cache_resource
def load_ecg_model():
    try:
        model = load_model("ecg_cnn_lstm_model.keras")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info("Make sure 'ecg_cnn_lstm_model.keras' is in the root folder of your repository.")
        return None

# ====================== Main App ======================
st.title("❤️ ECG Arrhythmia Detection System")
st.write("Upload a CSV file containing **1000** ECG signal values.")

# Load model
model = load_ecg_model()

if model is None:
    st.stop()

uploaded_file = st.file_uploader(
    "Choose ECG CSV File",
    type=["csv"],
    help="Upload a CSV file with exactly 1000 rows and 1 column (no header)"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file, header=None)
        
        # Flatten to 1D array
        ecg_signal = data.values.flatten()
        
        # Convert to float32
        ecg_signal = ecg_signal.astype(np.float32)

        # Validate length
        if len(ecg_signal) != 1000:
            st.error(f"❌ Expected 1000 values but got {len(ecg_signal)} values.")
            st.stop()

        # Reshape for model input: (batch_size, timesteps, features)
        input_signal = ecg_signal.reshape(1, 1000, 1)

        # Make prediction
        with st.spinner("Analyzing ECG..."):
            prediction = model.predict(input_signal, verbose=0)

        # Handle prediction output
        if prediction.shape[-1] == 1:  # Binary classification
            probability = float(prediction[0][0])
            is_abnormal = probability > 0.5
        else:  # Multi-class classification
            probability = float(np.max(prediction[0]))
            is_abnormal = np.argmax(prediction[0]) != 0  # Assuming 0 = Normal

        # ====================== Results ======================
        st.subheader("Prediction Result")

        if is_abnormal:
            st.error("⚠️ **Abnormal ECG Detected**")
        else:
            st.success("✅ **Normal ECG Detected**")

        st.write(f"**Confidence Score:** `{probability:.4f}`")

        # Plot the ECG signal
        st.subheader("ECG Signal Preview")
        st.line_chart(ecg_signal, use_container_width=True)

        # Optional: Show raw values
        with st.expander("Show Raw Signal Values (First 50)"):
            st.write(ecg_signal[:50])

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("Please make sure your CSV contains only numeric values and has exactly 1000 entries.")
