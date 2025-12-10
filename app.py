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

# Page config
st.set_page_config(
    page_title="BrainTalk - EEG to Speech",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Us", "FAQ & Resources"])

# Header - always show the header
st.image(IMAGE_ADDRESS, caption="EEG to Speech Conversion")

def load_model():
    try:
        model_path = Path(__file__).parent / "best_XGBoost_reg"
        if not model_path.exists():
            st.error(f"Model file does not exist at: {model_path}")
            return None
        if not os.access(str(model_path), os.R_OK):
            st.error(f"No read permission for the file: {model_path}")
            return None
        return joblib.load(str(model_path))
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
    return impute(extracted_features)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def download_edf_file(url, filename):
    try:
        # Create a session to handle cookies
        session = requests.Session()
        
        # First request to get the confirmation token
        response = session.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Save the content to a file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        
        # Verify the file was downloaded and has content
        if os.path.getsize(filename) > 0:
            # Try to read the file to verify it's a valid EDF
            try:
                raw = mne.io.read_raw_edf(filename, preload=False, verbose=False)
                return True
            except Exception as e:
                st.error(f"Invalid EDF file: {e}")
                os.remove(filename)  # Remove invalid file
                return False
        else:
            st.error("Downloaded file is empty")
            return False
            
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up if file was partially downloaded
        return False

def show_home_page():
    st.title("BrainTalk")
    
    # Create a directory for downloaded files if it doesn't exist
    os.makedirs("downloaded_edf", exist_ok=True)
    
    # Sample EDF files download section
    st.subheader("Download sample EDF files")
    st.write("Click on the buttons below to download sample EDF files for testing:")
    
    # Dictionary of sample EDF files with their Google Drive links
    edf_links = {
        'A': 'https://drive.google.com/uc?export=download&id=1ckD6gt7Z_Lkttg6kUv90ZbLQlanLP6NA',
        'C': 'https://drive.google.com/uc?export=download&id=1otwd0q5RWLbZZSW3BT7Fnt06cdMZO9FA',
        'F': 'https://drive.google.com/uc?export=download&id=1TGfrtfbLxvOlhQZrN-quUQmyP30X2rMh',
        'H': 'https://drive.google.com/uc?export=download&id=1MGTMQTeZXIvWEwrZM5GOoc0tlA-Ahqsg',
        'J': 'https://drive.google.com/uc?export=download&id=1revVfd-cHpLdvvyogTQdBGaNaosYUDpV',
        'M': 'https://drive.google.com/uc?export=download&id=1tc4Bv1Si11FFI_KsGWOnqTXvO6GEll61',
        'P': 'https://drive.google.com/uc?export=download&id=1ODZ0mc2LdHAG-BtmPYx8gyAasuhLlJN-',
        'S': 'https://drive.google.com/uc?export=download&id=1PsHjwSRjchDysKEpMv4QCH7VF6VuQkR6',
        'T': 'https://drive.google.com/uc?export=download&id=1lPiy9bhZ9bSQcW75wlZl6WUNEU9yH8Cw',
        'Y': 'https://drive.google.com/uc?export=download&id=1rlFXtwMHK1tJjsetPfFa42cVo0xMn1ea'
    }
    
    # Create a row of download buttons
    cols = st.columns(5)
    for i, (letter, url) in enumerate(edf_links.items()):
        with cols[i % 5]:
            if st.button(f"Download '{letter}'", 
                        key=f"btn_{letter}",
                        use_container_width=True):
                filename = os.path.join("downloaded_edf", f"sample_{letter}.edf")
                with st.spinner(f"Downloading {letter}.edf..."):
                    if download_edf_file(url, filename):
                        st.success(f"Downloaded {os.path.basename(filename)}")
                        # Create a download button for the downloaded file
                        with open(filename, "rb") as f:
                            st.download_button(
                                label=f"Save {letter}.edf",
                                data=f,
                                file_name=f"sample_{letter}.edf",
                                mime="application/octet-stream",
                                key=f"dl_{letter}"
                            )
                    else:
                        st.error("Download failed. Please try again.")
    # Upload multiple EDF files
    st.subheader("Upload your EEG data")
    uploaded_files = st.file_uploader("Select EDF files (in desired order)", 
                                    type="edf", 
                                    accept_multiple_files=True,
                                    key="file_uploader")
    st.markdown("---")
    
    # Load the model
    model = load_model()
    
    # Process uploaded files
    if uploaded_files:
        if st.button("Process EEG Data", type="primary"):
            with st.spinner("Processing EEG data..."):
                label_mapping = {0: 'A', 1: 'C', 2: 'F', 3: 'H', 4: 'J', 
                                5: 'M', 6: 'P', 7: 'S', 8: 'T', 9: 'Y'}
                all_labels = []

                for i, uploaded_file in enumerate(uploaded_files):
                    # Save the uploaded file
                    with open(f"temp_{i}.edf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        features_df = extract_eeg_features(f"temp_{i}.edf")
                        class_indices = model.predict(features_df)
                        unique, counts = np.unique(class_indices, return_counts=True)
                        most_common_index = np.argmax(counts)
                        most_common_element = unique[most_common_index]
                        all_labels.append(label_mapping[most_common_element])
                        
                        # Clean up the temporary file
                        os.remove(f"temp_{i}.edf")
                        
                    except Exception as e:
                        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                        continue

                if all_labels:
                    concatenated_labels = ''.join(all_labels)
                    st.subheader("Voice of the Mind")
                    st.write(concatenated_labels)
                    text_to_speech(concatenated_labels)
    
    # Show content based on selected page
if page == "About Us":
    st.title("About BrainTalk")
    st.markdown("""
    ## Welcome to BrainTalk
    
    BrainTalk is an innovative application that converts EEG brainwave data into spoken words, 
    providing a voice to individuals with speech impairments such as ALS, locked-in syndrome, 
    or other conditions affecting speech.
    
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
    
    ### How It Works
    1. Upload your EEG data in EDF format
    2. Our AI model processes the brainwave patterns
    3. The system predicts the intended characters
    4. Characters are combined to form words
    5. The text is converted to natural-sounding speech
    
    ### Getting Started
    Click on the 'Home' tab in the sidebar to begin converting your EEG data to speech.
    """)
    st.markdown("---")
    st.markdown("© 2025 BrainTalk - Empowering Communication Through Technology")

elif page == "FAQ & Resources":
    st.title("Frequently Asked Questions & Resources")
    
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
    
    st.markdown("---")
    st.markdown("Have more questions? Contact us at support@braintalk.example.com")

else:  # Home page
    show_home_page()
