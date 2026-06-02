import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load model with error handling
@st.cache_resource
def load_ecg_model():
    try:
        model = tf.keras.models.load_model("ecg_cnn_lstm_model.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

st.title("ECG Arrhythmia Detection System")
st.write("Upload a CSV file containing 1000 ECG signal values.")

model = load_ecg_model()

if model is None:
    st.stop()

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
        
        # Check data type and convert to float
        ecg_signal = ecg_signal.astype(np.float32)
        
        # Normalize signal (add if your model expects normalized input)
        # ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        if len(ecg_signal) != 1000:
            st.error(
                f"Expected 1000 values but found {len(ecg_signal)} values."
            )
        else:
            # Reshape for model (batch_size, timesteps, features)
            ecg_signal = ecg_signal.reshape(1, 1000, 1)
            
            # Prediction
            prediction = model.predict(ecg_signal, verbose=0)
            
            # Handle different output shapes
            if prediction.shape[-1] == 1:  # Binary classification
                probability = float(prediction[0][0])
                is_abnormal = probability > 0.5
            else:  # Multi-class
                probability = float(np.max(prediction[0]))
                is_abnormal = np.argmax(prediction[0]) != 0  # Assuming class 0 is normal
            
            st.subheader("Prediction Result")
            
            if is_abnormal:
                st.error("⚠️ Abnormal ECG Detected")
            else:
                st.success("✅ Normal ECG Detected")
            
            st.write(f"Confidence Score: {probability:.4f}")
            
            # Display signal plot
            st.subheader("ECG Signal Preview")
            st.line_chart(ecg_signal[0, :, 0])
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please ensure your CSV file contains numeric ECG values.")
