import os
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import azure.cognitiveservices.speech.translation as translation_sdk
import pyaudio
import wave

# Set up Azure Speech Translation configuration
#speech_key = "22c6c3ae0a6646beb757bf58f383e21f"
#service_region = "eastus"
#sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
 
  
speech_key = "d5ad9234f70742c7a5021d2b7b308031"
service_region = "eastus2"  # Use environment variable or secure storage in production

# Audio recording settings
class AudioRecorder:
    def __init__(self, duration=30, sample_rate=16000, channels=1, chunk=1024, device=0):
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.device = device
        self.frames = []

    def record(self):
        # Start recording
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device
        )

        st.write(f"Recording started for {self.duration} seconds")
        
        for _ in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)

        st.write("Recording Completed")
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the audio
        self.save()

    def save(self, filename="recorded_audio.wav"):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        st.write(f"Audio saved as {filename}")

# Function to output speech recognition result
def output_speech_recognition_result(translation_result):
    """Output the results of the speech recognition and translation."""
    if translation_result.reason == translation_sdk.ResultReason.TranslatedSpeech:
        st.write(f"**RECOGNIZED:** Text={translation_result.text}")
        for language, translation in translation_result.translations.items():
            st.write(f"**TRANSLATED into '{language}':** {translation}")
            #--------------------------------
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            # Note: the voice setting will not overwrite the voice element in input SSML.
            speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"

            text = translation

            # use the default speaker as audio output.
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

            result = speech_synthesizer.speak_text_async(text).get()
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesized for text [{}]".format(text))
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))


            #--------------------------------


    elif translation_result.reason == translation_sdk.ResultReason.NoMatch:
        st.write("**NOMATCH:** Speech could not be recognized.")
    elif translation_result.reason == translation_sdk.ResultReason.Canceled:
        cancellation = translation_result.cancellation_details
        st.write(f"**CANCELED:** Reason={cancellation.reason}")
        if cancellation.reason == translation_sdk.CancellationReason.Error:
            st.write(f"**CANCELED:** ErrorCode={cancellation.error_code}")
            st.write(f"**CANCELED:** ErrorDetails={cancellation.error_details}")
            st.write("**CANCELED:** Did you set the speech resource key and region values?")

# Function to recognize speech
def recognize_speech():
    """Perform speech recognition and translation."""
    audio_recorder = AudioRecorder(duration=30)  # Set your desired recording duration
    audio_recorder.record()  # Record audio

    # Use the recorded audio file for recognition
    audio_file = "recorded_audio.wav"
    speech_translation_config = translation_sdk.SpeechTranslationConfig(subscription=speech_key, region=service_region)
    speech_translation_config.speech_recognition_language = "en-US"
    speech_translation_config.add_target_language("ar")  # Change to desired target language (e.g., Arabic)

    audio_config = speechsdk.AudioConfig(filename=audio_file)  # Use recorded audio file
    translation_recognizer = translation_sdk.TranslationRecognizer(translation_config=speech_translation_config, audio_config=audio_config)

    st.write("Processing your audio...")
    translation_result = translation_recognizer.recognize_once()  # Use synchronous method here
    output_speech_recognition_result(translation_result)

    

# Streamlit UI layout
st.title("Azure Speech Translation with Streamlit")
st.write("Speak into your microphone, and it will be translated to your desired language.")

if st.button("Start Recording"):
    # Run the function to recognize speech
    recognize_speech()
