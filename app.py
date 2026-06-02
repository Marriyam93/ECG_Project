import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("ecg_cnn_lstm_model.keras")

st.title("ECG Arrhythmia Detection System")
st.write("Upload a CSV file containing 1000 ECG signal values.")

uploaded_file = st.file_uploader(
    "Choose ECG CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file, header=None)

        # Convert to numpy array
        ecg_signal = data.values.flatten()

        if len(ecg_signal) != 1000:
            st.error(
                f"Expected 1000 values but found {len(ecg_signal)} values."
            )
        else:
            # Reshape for model
            ecg_signal = ecg_signal.reshape(1, 1000, 1)

            # Prediction
            prediction = model.predict(ecg_signal)

            probability = float(prediction[0][0])

            st.subheader("Prediction Result")

            if probability > 0.5:
                st.error("Abnormal ECG Detected")
            else:
                st.success("Normal ECG Detected")

            st.write(f"Prediction Probability: {probability:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
