import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.image import resize
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go

classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS 
st.markdown("""
    <style>
    @media (prefers-color-scheme: dark) {
        .stMetric { background-color: #1e1e1e; border: 1px solid #404040; }
        [data-testid="stMetricValue"] { color: #70d5ff }
        [data-testid="stMetricLabel"] { color: #e0e0e0 }
    }
    .css-1d391kg { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource

#load the trained model
def load_model():
    mode_path = "./model_v2.keras"
    model = tf.keras.models.load_model(mode_path)
    return model
model = load_model()

#load and preprocess audio file
def load_and_preprocess_audio(file_path, target_shape=(150,150)):
    data =[]
    audio_data, sample_rate = librosa.load(file_path, sr= 44100)
    #performing preprocessing
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # duration of each chunk in seconds
    overlap_duration = 2  # overlap duration in seconds
     #convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    #calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples)))+1
    #interate over the chunks
    for i in range(num_chunks):
        #calculate the start and end sample indices for the current chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        #extra the chunk audio
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        #convert to decibels (log scale)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        #resize matrix base on provide target shape and add 1 channel
        mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=-1)
        mel_spectrogram_db_resize = resize(mel_spectrogram_db, target_shape)
        #append data to list
        data.append(mel_spectrogram_db_resize)
    return np.array(data)  

#predict
def predict_genre(model,x_test):
    y_pred = model.predict(x_test, verbose=0)  
    mean_probs = np.mean(y_pred, axis=0)        
    pred_class_idx = np.argmax(mean_probs)    
    return pred_class_idx,mean_probs

## Main Page
def home():
    st.title("🎵 Music Genre Classifier")
    st.markdown(''' ## Welcome to the Music Genre Classification System! 🎶🎧''')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        This application uses a **Convolutional Neural Network (CNN)** trained on the GTZAN dataset
        to classify music audio files into **10 different genres**.
        **Supported Genres:**
        - 🎸 Blues
        - 🎻 Classical
        - 🤠 Country
        - 🕺 Disco
        - 🎤 Hip-Hop
        - 🎺 Jazz
        - 🤘 Metal
        - 🎉 Pop
        - 🌴 Reggae
        - 🎸 Rock                  
        ### How it works:
        1. **Upload** an audio file (MP3 or WAV format)
        2. **Preprocessing** converts audio to Mel-spectrogram features
        3. **Prediction** runs through trained CNN model
        4. **Results** show predicted genre with confidence scores
        ### Advantages of the project
        - **Accuracy:** The system utilizes the most modern deep learning models to accurately predict genres.
        - **User-friendly:** Simple and intuitive interface for a smooth user experience.
        - **Fast and efficient:** Get results quickly, allowing for faster music classification and discovery.
        """)
    with col2:
        st.info("""
        ### Quick Stats
        - **Model**: CNN (4 Conv blocks)
        - **Training Accuracy**: 99.7%
        - **Validation Accuracy**: 96.37%     
        - **Test Accuracy**: 96.03%
        - **Input Shape**: 150×150×1 (Mel-spectrograms)
        - **Sampling Rate**: 44.1 kHz
        """)
    st.divider()

#About Project
def about():
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Music Genre Classification

    This is a comprehensive machine learning application for classifying music into different genres
    using deep learning techniques. 
    Experts have been trying for a long time to understand sound and what differenciates one song from another. 
    How to visualize sound. What makes a tone different from another. This data hopefully can give the opportunity to do just that.

    #### Project Overview
    - **Type**: Supervised Classification
    - **Dataset**: GTZAN Genre Collection (10 genres, 100 songs each)
    - **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
    - **Technology**: TensorFlow, Librosa, Streamlit
    - **Deployment**: Streamlit Cloud

    #### Features
    1. Single file prediction with confidence scores
    2. Audio visualization (mel spectrogram)
    3. Comprehensive analytics 
    4. Model performance metrics
    5. Dataset statistics
    6. Downloadable results

    #### How to Use
    1. **Navigate** using the sidebar menu
    2. **Upload** audio files (MP3 or WAV)
    3. **View** predictions with accuracy
    4. **Analyze** visualization of result ratios using pie charts

    #### Supported Audio Formats
    - MP3 (MPEG Audio)
    - WAV (Waveform Audio)

    #### Limitations & Future Work
    - **Maximum Batch Size**: 1 file
    - **Recording**: No direct audio recording from the system yet
    - **Future**: Allows simultaneous loading of multiple files, enables audio recording, 
                real-time model monitoring, and custom model fine-tuning
        """)

#Prediction Page
def prediction():
    st.title("🎵 Audio Prediction")
    #create folder if not exist to save uploaded files
    if not os.path.exists("./music_test"):
        os.makedirs("./music_test")
    test = st.file_uploader("Choose an audio file", type=["mp3", "wav"],label_visibility="collapsed")
    if test is not None:
        # Save uploaded file to disk for processing
        filepath = "./music_test/" + test.name
        with open(filepath, "wb") as f:
            f.write(test.getbuffer())
        # Display audio player
        st.audio(test, format="audio/mp3" if test.name.endswith('.mp3') else "audio/wav")
    else:
        st.warning("Please upload an audio file to proceed with prediction.")
        return
    
     # Display results
    if(st.button("Predict")):
        with st.spinner("Analyzing audio..."):       
            audio_data = load_and_preprocess_audio(filepath)
            pred_class_idx, mean_probs = predict_genre(model,audio_data)

        st.subheader("**Predicted Genre: **:red[{}]** " \
        "with percentage: :red[{:.2f}%]**".format(classes[pred_class_idx].upper(), mean_probs[pred_class_idx]*100))
        st.divider()
        
        # Visualizations
        st.subheader("Confidence Distribution")
        #post data into pandas dataframe for plotly better read
        df = pd.DataFrame({
            'Genre': classes,
            'Confidence (%)': (mean_probs*100).round(2)
            })
        #sort dataframe to show the most confident genre at the top
        df = df.sort_values(by='Confidence (%)', ascending=False)
        #create pie chart using plotly express
        fig = px.pie(df, names='Genre', values='Confidence (%)', 
                     title='Genre Prediction Confidence Breakdown',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
              
        # Audio visualizations
        #show processed mel-spectrogram from audio_data[0]    
        st.subheader("Mel-Spectrogram (first 5 Chunk)")
        for i in range(0,5):
            fig = plt.figure(figsize=(10,4))
            mel_spec = audio_data[i].squeeze() #remove channel dimension
            librosa.display.specshow(mel_spec, sr=44100, x_axis='time', y_axis='mel')
            plt.colorbar(format='%2.0f dB')
            plt.tight_layout()
            st.pyplot(fig)
                
def analytics():
    st.title("📊 Model Analytics & Dataset Info")
    # Get information
    model_summary = {
        "input_shape": model.input_shape,
        "training_accuracy": 0.997,
        "validation_accuracy": 0.9637,
        "test_accuracy": 0.9603,
    }
    # Model Information
    st.subheader("🤖 Model Architecture")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Shape", f"{model_summary['input_shape'][1:]}")
    with col2:
        st.metric("Number of Genres", 10)
    with col3:
        st.metric("Sampling Rate", f"44100Hz")
    with col4:
        st.metric("Audio Chunk Size", f"4s")
    st.divider()

    # Model Performance
    st.subheader("📈 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Accuracy", f"{model_summary['training_accuracy']:.2%}")
    with col2:
        st.metric("Validation Accuracy", f"{model_summary['validation_accuracy']:.2%}")
    with col3:
        st.metric("Test Accuracy", f"{model_summary['test_accuracy']:.2%}")
    st.divider()

    #Create bar chart comparing train/val/test accuracy.
    st.subheader("📊 Accuracy Comparison")
    accuracies = [model_summary['training_accuracy'],model_summary['validation_accuracy'],model_summary['test_accuracy']]
    fig = go.Figure(data=[go.Bar(x=['Training', 'Validation', 'Test'],
                                  y=accuracies,
                                  marker=dict(color=['#2ecc71', '#3498db', '#e74c3c']),
                                  text=[f'{acc:.2%}' for acc in accuracies],
                                  textposition='auto')])
    fig.update_layout(title="Model Performance Across Stages", yaxis_title="Accuracy",
                        xaxis_title="Stage", height=400, showlegend=False, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Info
    st.subheader("🎵 Dataset Information")
    st.info(f"""
        **Dataset**: GTZAN Music Genre Classification
        - **Total Genres**: 10, each with 100 audio files (30 seconds each)
        - **Genres Supported**: 🎸 Blues, 🎻 Classical,🤠 Country,🕺 Disco,🎤 Hip-Hop,
            🎺 Jazz,🤘 Metal, 🎉 Pop,🌴 Reggae, 🎸 Rock
        """)
    st.divider()

    # About the model
    st.subheader("ℹ️ About This System")
    st.markdown("""
    ### Architecture
    - **Type**: Convolutional Neural Network (CNN)
    - **Framework**: TensorFlow/Keras
    - **Input**: 150×150×1 Mel-spectrograms
    - **Layers**: 4 Conv2D blocks with BatchNormalization
    - **Regularization**: Dropout (0.3-0.5), Early Stopping

    ### Processing Pipeline
    1. **Audio Loading**: Librosa (44.1 kHz sampling rate)
    2. **Chunking**: 4-second chunks with 2-second overlap
    3. **Feature Extraction**: Mel-spectrogram (128 bands)
    4. **Preprocessing**: Zero-centered normalization
    5. **Prediction**: Average across chunks

    ### Model Training
    - **Optimizer**: Adam (learning rate: 0.0001)
    - **Loss**: Categorical Cross-Entropy
    - **Metrics**: Accuracy
    - **Data Split**: 60% training, 20% validation, 20% test
    - **Augmentation**: Horizontal flip
    """)

# Main app logic
def main():
    # Sidebar navigation
    st.sidebar.title("🎵 Navigation")
    page = st.sidebar.radio("Select Page",["Home","Prediction","Analytics","About Project"], label_visibility="collapsed")
    # Render selected page
    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    elif page == "Analytics":
        analytics()
    elif page == "About Project":
        about()
    # Footer
    st.sidebar.divider()
if __name__ == "__main__":
    main()
