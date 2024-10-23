
speech_key = "22c6c3ae0a6646beb757bf58f383e21f"
service_region = "eastus"

import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import pyaudio
import wave
from io import BytesIO

# Set up Azure Speech configurati
class AudioRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024, device=None, duration=5):
        self.format = format
        self.channels = channels
        self.sample_rate = rate
        self.chunk = chunk
        self.device = device
        self.duration = duration
        self.audio = None
        self.frames = []

    def record(self):
        # start Recording
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device)
        
        st.write(f"Recording started for {self.duration} seconds")
        self.frames = []
        
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)

        st.write("Recording Completed")
        
        # stop Recording
        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        self.save()

    def save(self):
        output = BytesIO()
        wave_file = wave.open(output, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.audio.get_sample_size(self.format))
        wave_file.setframerate(self.sample_rate)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()
        output.seek(0)  # Reset pointer to start
        return output

def azure_speech_to_text(audio_data):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_input = speechsdk.AudioConfig(filename=audio_data)  # Using filename instead of stream
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = recognizer.recognize_once()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech recognized."
    else:
        return f"Speech recognition error: {result.reason}"

# Streamlit UI layout
st.title("Azure Speech-to-Text with Streamlit")
st.write("Record your voice and convert it to text using Azure Speech API.")

# Initialize audio recorder
recorder = AudioRecorder(duration=5)

if st.button("Start Recording"):
    recorder.record()
    audio_bytes = recorder.save()

    # Save audio to a file to pass to Azure
    audio_file_path = "temp_audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio_bytes.getvalue())

    st.audio(audio_bytes.getvalue(), format="audio/wav")
    
    # Call Azure Speech-to-Text API
    text_output = azure_speech_to_text(audio_file_path)
    st.write("**Transcribed Text:**", text_output)
