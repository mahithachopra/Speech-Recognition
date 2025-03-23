import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import os
import librosa
import soundfile as sf

# Load pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(audio_file):
    speech, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

st.title("Speech Recognition")
st.write("Upload an audio file for speech-to-text transcription.")

# Option to upload an audio file
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "flac", "mp3"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    st.audio(temp_audio_path, format='audio/wav')
    st.write("Transcribing...")
    transcription = transcribe_audio(temp_audio_path)
    st.write("**Transcription:**")
    st.text_area("", transcription, height=100)
    
    # Provide download option for transcription
    st.download_button(label="Download Transcription", data=transcription, file_name="transcription.txt", mime="text/plain")
    os.remove(temp_audio_path)

