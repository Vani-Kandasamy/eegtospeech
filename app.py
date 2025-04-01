import streamlit as st
import mne
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import joblib
from pathlib import Path

from gtts import gTTS
from io import BytesIO


import os

IMAGE_ADDRESS = "https://www.tsukuba.ac.jp/en/research-news/images/p20230904180000.jpg"



st.image(IMAGE_ADDRESS, caption = "EEG to Speech")

def load_model():
    try:
        model_path = Path(__file__).parent / "best_XGBoost_reg"
        #st.write(f"Checking model path: {model_path}")

        if not model_path.exists():
            st.error(f"Model file does not exist at: {model_path}")
            return None

        if not os.access(str(model_path), os.R_OK):
            st.error(f"No read permission for the file: {model_path}")
            return None

        model = joblib.load(str(model_path))
        #st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

def text_to_speech(text):
    try:
        # Convert text to speech
        tts = gTTS(text=text, lang='en')

        # Save to a bytes buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)

        # Reset buffer position to start
        audio_buffer.seek(0)

        # Play the audio using Streamlit
        st.audio(audio_buffer, format="audio/mp3")

    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    # web app
    st.title("BrainTalk")

    # Upload multiple EDF files
    uploaded_files = st.file_uploader("Upload EEG EDF files (select in desired order)", type="edf", accept_multiple_files=True)
    text = ''
    # Load the model
    model = load_model()
    # Check if there are any uploaded files
    if uploaded_files:
        if st.button("Express Waves"):
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
            st.subheader("Voice of the Mind")
            st.write(concatenated_labels)
            text = concatenated_labels
        #st.write(text)
        if text:
            text_to_speech(text)

if __name__ == "__main__":
    main()