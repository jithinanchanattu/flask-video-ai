from flask import Flask, jsonify
import os
import streamlit as st
import subprocess
from TTS.api import TTS
from googletrans import Translator
from google.colab import files

app = Flask(__name__)


@app.route('/')
def index():
    main()

# Define language mapping
language_mapping = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Malayalam': 'ml',
    'Hindi': 'hi',
    'Arabic': 'ar',
    'Chinese (Simplified)': 'zh-cn'
}

def upload_video():
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        return "uploaded_video.mp4"
    return None

def resize_video(filename):
    output_filename = f"resized_{filename}"
    cmd = f"ffmpeg -i {filename} -vf scale=-1:720 {output_filename}"
    subprocess.run(cmd, shell=True)
    return output_filename

def extract_text(video_path):
    ffmpeg_command = f"ffmpeg -i '{video_path}' -acodec pcm_s24le -ar 48000 -q:a 0 -map a -y 'output_audio.wav'"
    subprocess.run(ffmpeg_command, shell=True)
    model = whisper.load_model("base")
    result = model.transcribe("output_audio.wav")
    return result["text"]

def translate_text(text, target_language_code):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language_code).text
    return translated_text

def synthesize_audio(text, language):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to("cuda")
    tts.tts_to_file(text, speaker_wav='output_audio.wav', file_path="output_synth.wav", language=language)
    return "output_synth.wav"

def download_audio_video(audio_path, video_path):
    st.audio(audio_path, format='audio/wav')
    st.warning("Downloading video and audio...")
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    st.download_button(label="Download Video", data=video_bytes, file_name="output_video.mp4")
    st.download_button(label="Download Audio", data=audio_bytes, file_name="output_audio.wav")

def main():
    st.title("Video to Audio Conversion and Translation App")

    # Upload Video
    st.header("Step 1: Upload Video")
    video_path = upload_video()

    if video_path:
        st.success("Video uploaded successfully.")
    else:
        st.warning("Please upload a video to proceed.")

    # Resize Video (Optional)
    resize_to_720p = st.checkbox("Resize to 720p (better results)")
    if resize_to_720p and video_path:
        st.info("Resizing to 720p...")
        video_path = resize_video(video_path)

    # Extract Audio Text From Video - Whisper
    if st.button("Extract Audio Text"):
        if video_path:
            extracted_text = extract_text(video_path)
            st.success("Audio text extracted successfully.")
            st.text("Audio Text:")
            st.write(extracted_text)
        else:
            st.warning("Please upload a video first.")

    # Translation
    st.header("Step 3: Translation")
    if st.button("Translate Text"):
        if 'extracted_text' in st.session_state:
            target_language = st.selectbox("Select target language", list(language_mapping.keys()))
            translated_text = translate_text(st.session_state['extracted_text'], language_mapping[target_language])
            st.success("Text translated successfully.")
            st.text("Translated Text:")
            st.write(translated_text)
        else:
            st.warning("Please extract audio text first.")

    # Voice Synthesis
    st.header("Step 4: Voice Synthesis")
    if st.button("Synthesize Audio"):
        if 'translated_text' in st.session_state:
            synthesized_audio_path = synthesize_audio(st.session_state['translated_text'], language_mapping[target_language])
            st.success("Audio synthesized successfully.")
            st.audio(synthesized_audio_path, format='audio/wav')
            st.success("Audio played successfully.")
        else:
            st.warning("Please translate text first.")

    # Download Audio and Video
    st.header("Step 5: Download Audio and Video")
    if st.button("Download Audio and Video"):
        if 'synthesized_audio_path' in st.session_state and video_path:
            download_audio_video(st.session_state['synthesized_audio_path'], video_path)
        else:
            st.warning("Please synthesize audio and upload video first.")

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
