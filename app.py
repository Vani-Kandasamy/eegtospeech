import streamlit as st
import mne
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import joblib

# Load the pre-trained model
MODEL_PATH = '/workspaces/eeg_speech/best_XGBoost_reg'
model = joblib.load(MODEL_PATH)

def extract_eeg_features(edf_path):
    # Read the EDF file using MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Extract data and times
    data, times = raw.get_data(return_times=True)

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

    # Impute missing values
    extracted_features = impute(extracted_features)

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

        st.write(features_df.shape)
        # Example mapping: assuming these were your labels
        label_mapping = {0: 'A', 1: 'C', 2:'F',3: 'H', 4:'J',5: 'M',6: 'P',7: 'S', 8: 'T',9: 'Y'}

        # Make prediction using the pre-trained model
        class_indices = model.predict(features_df)

        #Take majority as class prediction
        class_indices = max(set(class_indices), key=class_indices.count)

        # Map indices to actual labels
        actual_label = [label_mapping[index] for index in class_indices]


        st.subheader("Model Prediction")
        st.write(actual_label)

if __name__ == "__main__":
    main()