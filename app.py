import streamlit as st
import mne
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import joblib

from gtts import gTTS

# Load the pre-trained model
MODEL_PATH = '/workspaces/eeg_speech/best_XGBoost_reg'
model = joblib.load(MODEL_PATH)

def extract_eeg_features(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data, times = raw.get_data(return_times=True)
    df_list = []

    for i, channel_name in enumerate(raw.ch_names):
        df = pd.DataFrame({
            'id': i,
            'time': times,
            'value': data[i]
        })
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    extracted_features = extract_features(full_df, column_id='id', column_sort='time', column_value='value')
    extracted_features = impute(extracted_features)

    return extracted_features

def text_to_speech(text, output_file):
    # Create a gTTS object
    tts = gTTS(text=text, lang='en')

    # Save the audio to a file
    tts.save(output_file)
    return output_file

def main():
    st.title("EEG Feature Extraction and Prediction App")

    # Upload multiple EDF files
    uploaded_files = st.file_uploader("Upload EEG EDF files (select in desired order)", type="edf", accept_multiple_files=True)

    # Check if there are any uploaded files
    if uploaded_files:
        if st.button("Start Feature Extraction and Prediction"):
            label_mapping = {0: 'A', 1: 'C', 2: 'F', 3: 'H', 4: 'J', 5: 'M', 6: 'P', 7: 'S', 8: 'T', 9: 'Y'}
            all_labels = []

            for i, uploaded_file in enumerate(uploaded_files):
                with open(f"temp_{i}.edf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                features_df = extract_eeg_features(f"temp_{i}.edf")
                class_indices = model.predict(features_df)
                unique, counts = np.unique(class_indices, return_counts=True)
                most_common_index = np.argmax(counts)
                most_common_element = unique[most_common_index]
                actual_label = label_mapping[most_common_element]
                all_labels.append(actual_label)

            concatenated_labels = ''.join(all_labels)
            st.subheader("Concatenated Model Prediction")
            st.write(concatenated_labels)
       
            # Button to trigger the text-to-speech conversion
            if st.button("Convert and Play"):
                output_file = "output.mp3"
                output_file = text_to_speech(concatenated_labels, output_file)

                # Use Streamlit's audio component to play the audio file
                audio_file = open(output_file, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()