import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.get_logger().setLevel('ERROR')

@st.cache_resource
def load_ecg_model():
    try:
        model = load_model("ecg_cnn_lstm_model.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

st.title("ECG Arrhythmia Detection System")
st.write("Upload a CSV file containing 1000 ECG signal values.")

model = load_ecg_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose ECG CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)
        ecg_signal = data.values.flatten().astype(np.float32)

        if len(ecg_signal) != 1000:
            st.error(f"Expected 1000 values but found {len(ecg_signal)} values.")
        else:
            ecg_signal = ecg_signal.reshape(1, 1000, 1)
            
            prediction = model.predict(ecg_signal, verbose=0)
            
            # Binary or Multi-class handling
            if prediction.shape[-1] == 1:
                prob = float(prediction[0][0])
                is_abnormal = prob > 0.5
            else:
                prob = float(np.max(prediction[0]))
                is_abnormal = np.argmax(prediction[0]) != 0

            st.subheader("Prediction Result")
            if is_abnormal:
                st.error("⚠️ Abnormal ECG Detected")
            else:
                st.success("✅ Normal ECG Detected")
            
            st.write(f"Confidence Score: {prob:.4f}")
            st.subheader("ECG Signal Preview")
            st.line_chart(ecg_signal[0, :, 0])

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
