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
    data, times = raw[:]
    ch_names = raw.ch_names
    df = pd.DataFrame(data.T, columns=ch_names)
    df['time'] = times
    '''
    # Reset the index for tsfresh compatibility (e.g., ensuring time sequential data)
    df_melted = df.melt(id_vars=['time'], var_name='channel', value_name='amplitude')

    # Add dummy id column (necessary for tsfresh, typically based on user-specific requirements)
    df_melted['id'] = 1  # Assuming only one signal for this example
    '''
    # Initialize an empty list for storing DataFrames
    df_list = []

    # Iterate over each channel
    for i, channel_name in enumerate(raw.ch_names):
        # Create a DataFrame for each channel
        df = pd.DataFrame({
            'id': i,  # Assign a unique ID to each channel
            'time': times,
            'value': data[i]
        })
        df_list.append(df)

    # Concatenate all channel DataFrames into a single one
    full_df = pd.concat(df_list, ignore_index=True)
    # Extract features using tsfresh
    extracted_features = extract_features(full_df, column_id='id', column_sort='time', column_value='value')

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

        # Example mapping: assuming these were your labels
        label_mapping = {0: 'A', 1: 'C', 2:'F',3: 'H', 4:'J',5: 'M',6: 'P',7: 'S', 8: 'T',9: 'Y'}

        # Make prediction using the pre-trained model
        class_indices = model.predict(features_df)

        # Map indices to actual labels
        actual_labels = [label_mapping[index] for index in class_indices]


        st.subheader("Model Prediction")
        st.write(actual_labels)

if __name__ == "__main__":
    main()