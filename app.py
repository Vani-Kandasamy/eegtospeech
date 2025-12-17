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
import requests

# Constants
IMAGE_ADDRESS = "https://www.tsukuba.ac.jp/en/research-news/images/p20230904180000.jpg"

# Initialize session state
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

# Load model
def load_model():
    try:
        model_path = Path(__file__).parent / "best_XGBoost_reg"
        if not model_path.exists():
            st.error(f"Model file does not exist at: {model_path}")
            return None
        return joblib.load(str(model_path))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Extract EEG features
def extract_eeg_features(edf_path):
    try:
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
        return impute(extracted_features)
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3")
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")

def process_edf_file(uploaded_file, model, label_mapping):
    """Process a single EDF file and return the predicted character."""
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the file
        features_df = extract_eeg_features(temp_path)
        if features_df is not None:
            # Make prediction
            class_indices = model.predict(features_df)
            unique, counts = np.unique(class_indices, return_counts=True)
            most_common_index = np.argmax(counts)
            predicted_letter = label_mapping[unique[most_common_index]]
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return predicted_letter, True
        return None, False
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None, False

def show_home_page():
    st.title("BrainTalk - EEG to Speech Conversion")
    st.image(IMAGE_ADDRESS, use_column_width=True)
    
    st.markdown("""
    ### Upload multiple EDF files to form words
    
    Upload EDF files in order - each file represents one character in the word.
    The app will process them sequentially and combine the results.
    """)
    
    # File uploader - accepts multiple files
    uploaded_files = st.file_uploader(
        "Select EDF files (in order)", 
        type=["edf"],
        accept_multiple_files=True,
        help="Select multiple EDF files in the correct order"
    )
    
    if uploaded_files and st.button("Process EEG Data", type="primary"):
        with st.spinner("Processing your EEG data..."):
            # Load model and label mapping
            model = load_model()
            if not model:
                return
                
            label_mapping = {0: 'A', 1: 'C', 2: 'F', 3: 'H', 4: 'J', 
                           5: 'M', 6: 'P', 7: 'S', 8: 'T', 9: 'Y'}
            
            # Process each file
            predicted_chars = []
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing file {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
                
                # Process the file
                predicted_char, success = process_edf_file(uploaded_file, model, label_mapping)
                if success:
                    predicted_chars.append(predicted_char)
                    st.write(f"File {i+1}: Predicted '{predicted_char}'")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Combine predictions and show results
            if predicted_chars:
                predicted_word = ''.join(predicted_chars)
                st.success("Processing complete!")
                st.subheader("Predicted Word")
                st.markdown(f"## {predicted_word}")
                
                # Generate and play audio
                st.subheader("Listen to the result")
                text_to_speech(predicted_word)
            else:
                st.warning("No valid predictions were made from the uploaded files.")

def show_sample_files_page():
    st.title("Sample EDF Files")
    st.write("Click on any sample EDF file to process it. You can select multiple files to form words.")
    
    sample_files = {
        "A": "https://drive.google.com/uc?export=download&id=1ckD6gt7Z_Lkttg6kUv90ZbLQlanLP6NA",
        "C": "https://drive.google.com/uc?export=download&id=1otwd0q5RWLbZZSW3BT7Fnt06cdMZO9FA",
        "F": "https://drive.google.com/uc?export=download&id=1TGfrtfbLxvOlhQZrN-quUQmyP30X2rMh",
        "H": "https://drive.google.com/uc?export=download&id=1MGTMQTeZXIvWEwrZM5GOoc0tlA-Ahqsg",
        "J": "https://drive.google.com/uc?export=download&id=1revVfd-cHpLdvvyogTQdBGaNaosYUDpV",
        "M": "https://drive.google.com/uc?export=download&id=1tc4Bv1Si11FFI_KsGWOnqTXvO6GEll61",
        "P": "https://drive.google.com/uc?export=download&id=1ODZ0mc2LdHAG-BtmPYx8gyAasuhLlJN-",
        "S": "https://drive.google.com/uc?export=download&id=1PsHjwSRjchDysKEpMv4QCH7VF6VuQkR6",
        "T": "https://drive.google.com/uc?export=download&id=1lPiy9bhZ9bSQcW75wlZl6WUNEU9yH8Cw",
        "Y": "https://drive.google.com/uc?export=download&id=1rlFXtwMHK1tJjsetPfFa42cVo0xMn1ea"
    }
    
    # Display sample files with checkboxes
    st.subheader("Available Sample Files")
    selected_letters = []
    selection_order = {}
    cols = st.columns(5)
    
    for i, (letter, url) in enumerate(sample_files.items()):
        with cols[i % 5]:
            if st.checkbox(f"Letter '{letter}'", key=f"cb_{letter}"):
                if letter not in selection_order:
                    selection_order[letter] = len(selection_order)
                selected_letters.append((letter, url))
    
    # Sort selected_letters based on the order of selection
    selected_letters.sort(key=lambda x: selection_order[x[0]])
    
    # Process selected files
    if selected_letters and st.button("Process Selected Files", type="primary"):
        with st.spinner("Processing selected files..."):
            # Load model and label mapping
            model = load_model()
            if not model:
                return
                
            label_mapping = {0: 'A', 1: 'C', 2: 'F', 3: 'H', 4: 'J', 
                           5: 'M', 6: 'P', 7: 'S', 8: 'T', 9: 'Y'}
            
            predicted_chars = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (letter, url) in enumerate(selected_letters):
                # Update progress
                progress = (i + 1) / len(selected_letters)
                progress_bar.progress(progress)
                status_text.text(f"Processing letter {i+1} of {len(selected_letters)}: '{letter}'")
                
                try:
                    # Download the file
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Save temporarily
                    temp_path = f"temp_{letter}.edf"
                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Process the file
                    features_df = extract_eeg_features(temp_path)
                    if features_df is not None:
                        class_indices = model.predict(features_df)
                        unique, counts = np.unique(class_indices, return_counts=True)
                        most_common_index = np.argmax(counts)
                        predicted_letter = label_mapping[unique[most_common_index]]
                        predicted_chars.append(predicted_letter)
                        st.write(f"Sample '{letter}' → Predicted: {predicted_letter}")
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"Error processing sample '{letter}': {str(e)}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show combined results
            if predicted_chars:
                predicted_word = ''.join(predicted_chars)
                st.success("Processing complete!")
                st.subheader("Predicted Word")
                st.markdown(f"## {predicted_word}")
                
                # Generate and play audio
                st.subheader("Listen to the result")
                text_to_speech(predicted_word)
            else:
                st.warning("No valid predictions were made from the selected files.")


# About Page
def show_about_page():
    st.title("About BrainTalk")
    st.markdown("""
    ## Our Mission
    
    BrainTalk's mission is to provide a voice to those who have lost the ability to speak 
    due to neurological conditions, using cutting-edge EEG technology and machine learning.
    
    ### Key Features
    - **EEG to Text Conversion**: Upload EEG data in EDF format to convert brain signals into text
    - **Text to Speech**: Hear the converted text with our built-in speech synthesis
    - **User-Friendly Interface**: Simple and intuitive design for ease of use
    - **Privacy-Focused**: Your data stays on your device and is not stored on our servers
    
    ### Who Can Benefit
    - Individuals with ALS (Amyotrophic Lateral Sclerosis)
    - Patients with locked-in syndrome
    - People with speech disorders
    - Stroke survivors with speech impairments
    - Researchers in the field of Brain-Computer Interfaces (BCI)
    - Healthcare professionals working with non-verbal patients
    
    ## How It Works
    
    1. **EEG Data Collection**: Brain signals are recorded using non-invasive EEG headsets
    2. **Signal Processing**: Advanced algorithms process the raw EEG data
    3. **Machine Learning**: Our AI model interprets the brain signals
    4. **Text Generation**: The interpreted signals are converted to text
    5. **Speech Synthesis**: The text is spoken aloud using natural-sounding voices
    """)

# FAQ Page
def show_faq_page():
    st.title("Frequently Asked Questions")
    
    faq_expander = st.expander("General Questions", expanded=True)
    with faq_expander:
        st.markdown("""
        **Q: What is BrainTalk?**  
        A: BrainTalk converts EEG brainwave data into spoken words, helping individuals with speech impairments communicate.
        
        **Q: How accurate is the EEG to text conversion?**  
        A: Accuracy depends on EEG data quality and individual users. Our model is trained on diverse datasets, but results may vary.
        
        **Q: Is my data secure?**  
        A: Yes, all processing happens locally on your device. We don't store your EEG data.
        """)
    
    als_expander = st.expander("For ALS Patients and Caregivers")
    with als_expander:
        st.markdown("""
        **Q: How can ALS patients benefit from BrainTalk?**  
        A: BrainTalk provides a non-invasive communication method for ALS patients as their condition progresses.
        
        **Q: What equipment do I need?**  
        A: You'll need an EEG headset that exports data in EDF format. Consult your healthcare provider for recommendations.
        """)
    
    resources_expander = st.expander("Helpful Resources")
    with resources_expander:
        st.markdown("""
        ### Community Forums and Support Groups
        - [ALS Association Discussion Forums](https://www.als.org/community/discussion-forums)
        - [ALS Forums](https://www.alsforums.com/)
        - [Brain-Computer Interface Community](https://www.bci-info.org/)
        - [Reddit r/ALS](https://www.reddit.com/r/ALS/)
        - [Reddit r/BCI](https://www.reddit.com/r/BCI/)
        
        ### Research and Information
        - [ALS Association](https://www.als.org/)
        - [International Brain-Computer Interface Society](http://bcisociety.org/)
        - [National Institute of Neurological Disorders and Stroke](https://www.ninds.nih.gov/)
        """)
# Update the main function to include the new page
def main():
    init_session_state()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Sample Files", "About", "FAQ"])
    
    # Page Routing
    if page == "Home":
        show_home_page()
    elif page == "Sample Files":
        show_sample_files_page()
    elif page == "About":
        show_about_page()
    elif page == "FAQ":
        show_faq_page()

if __name__ == "__main__":
    main()
