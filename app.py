import streamlit as st
import mne
import pandas as pd
import numpy as np
from tsfresh import extract_features
import joblib

# Load the pre-trained model
MODEL_PATH = '/workspaces/eeg_speech/best_XGBoost_reg'
model = joblib.load(MODEL_PATH)

def extract_eeg_features(edf_path):
    # Read the EDF file using MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Get data and create a DataFrame
    data, times = raw.get_data(return_times=True)
    ch_names = raw.ch_names
    df = pd.DataFrame(data.T, columns=ch_names)
    df['time'] = times

    # Reset the index for tsfresh compatibility (e.g., ensuring time sequential data)
    df_melted = df.melt(id_vars=['time'], var_name='channel', value_name='amplitude')

    # Add dummy id column (necessary for tsfresh, typically based on user-specific requirements)
    df_melted['id'] = 1  # Assuming only one signal for this example

    # Extract features using tsfresh
    extracted_features = extract_features(df_melted, column_id='id', column_sort='time')

    return extracted_features

def main():
    st.title("EEG Feature Extraction and Prediction App")

    # Upload the EDF file
    uploaded_file = st.file_uploader("Upload an EEG EDF file", type="edf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract features from the uploaded EDF file
        features_df = extract_eeg_features("temp.edf")

        # Display extracted features
        st.subheader("Extracted Features")
        st.write(features_df)

        # Make prediction using the pre-trained model
        prediction = model.predict(features_df)

        st.subheader("Model Prediction")
        st.write(prediction)

if __name__ == "__main__":
    main()