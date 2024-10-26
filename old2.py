
import os
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import azure.cognitiveservices.speech.translation as translation_sdk
import wave

from dotenv import load_dotenv

load_dotenv()
speech_key = os.getenv("speech_key")
service_region = os.getenv("service_region")

# Function to output speech recognition result
def output_speech_recognition_result(translation_result):
    """Output the results of the speech recognition and translation."""
    if translation_result.reason == translation_sdk.ResultReason.TranslatedSpeech:
        st.write(f"**RECOGNIZED:** Text={translation_result.text}")
        for language, translation in translation_result.translations.items():
            st.write(f"**TRANSLATED into '{language}':** {translation}")

            # Synthesize the translated text to speech
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"

            text = translation
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            result = speech_synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                st.write(f"Speech synthesized for text [{text}]")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                st.write(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    st.write(f"Error details: {cancellation_details.error_details}")

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
def recognize_speech(audio_file):
    """Perform speech recognition and translation."""
    speech_translation_config = translation_sdk.SpeechTranslationConfig(subscription=speech_key, region=service_region)
    speech_translation_config.speech_recognition_language = "en-US"
    speech_translation_config.add_target_language("ar")  # Change to desired target language (e.g., Arabic)

    audio_config = speechsdk.AudioConfig(filename=audio_file)  # Use uploaded audio file
    translation_recognizer = translation_sdk.TranslationRecognizer(translation_config=speech_translation_config, audio_config=audio_config)

    st.write("Processing your audio...")
    translation_result = translation_recognizer.recognize_once()  # Use synchronous method here
    output_speech_recognition_result(translation_result)

# Streamlit UI layout
st.title("Azure Speech Translation with Streamlit")
st.write("Upload a WAV audio file, and it will be translated to your desired language.")

# File upload
uploaded_file = st.file_uploader("Choose a WAV file...", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Start Processing"):
        # Run the function to recognize speech using the uploaded file
        recognize_speech("uploaded_audio.wav")
