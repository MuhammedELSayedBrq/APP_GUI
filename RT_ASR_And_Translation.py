import streamlit as st
import webrtcvad
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from pydub.playback import play
import io
import multiprocessing as mp
from scipy.io import wavfile
import torch
import pyaudio
import sounddevice as sd
import numpy as np
from queue import Queue
import threading
import socket
from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsTokenizer, VitsModel, set_seed

import mlflow  

# terminal command: mlflow ui
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("speech-translation-app")  


def Record_thread(Receiving_Q, comm_q):
    p = pyaudio.PyAudio()
    IN_st = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=16000,
                   input=True,
                   frames_per_buffer=2048)
    OUT_st = sd.OutputStream(channels=1, dtype="int16", samplerate=16000)
    OUT_st.start()
    last_state = "stop"
    while True:
        if not comm_q.empty():
            last_state = comm_q.get()
            OUT_st.stop()
            OUT_st.close()
            if last_state == "dev_org":
                OUT_st = sd.OutputStream(
                    channels=1, dtype=np.int16, samplerate=16000)
                OUT_st.start()
        if last_state in ["dev_org", "dev"]:
            chunk = IN_st.read(2048)
            audio_chunk = np.frombuffer(chunk, dtype=np.int16)
            Receiving_Q.put(audio_chunk)
            if last_state == "dev_org":
                OUT_st.write(audio_chunk)

# Silero VAD Thread

def VAD_thread(Receiving_Q, IN_Speech_Q, Denum_Q):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    Speech_Chunk = np.array([], dtype=np.float32)
    Silence_Chunk = np.array([], dtype=np.float32)
    denum = 32767
    while True:
        if not Denum_Q.empty():
            denum = int(Denum_Q.get())
            Speech_Chunk = np.array([], dtype=np.float32)
            Silence_Chunk = np.array([], dtype=np.float32)

        if not Receiving_Q.empty():
            audio = Receiving_Q.get()
            audio = np.array(audio / denum, dtype=np.float32)
            for _ in range(4):
                frame = audio[0:512]
                audio = audio[512:]
                new_confidence = model(torch.from_numpy(frame), 16000).item()
                if new_confidence > 0.4:
                    Speech_Chunk = np.concatenate(
                        (Speech_Chunk, Silence_Chunk, frame))
                    Silence_Chunk = np.array([], dtype=np.float32)
                else:
                    Silence_Chunk = np.concatenate((Silence_Chunk, frame))
                    if len(Silence_Chunk) > 8000:
                        if len(Speech_Chunk) > 12800:
                            Speech_Chunk = np.concatenate(
                                (Speech_Chunk, Silence_Chunk))
                            IN_Speech_Q.put(Speech_Chunk)
                            print("detect_sentence-------------------------------")
                        Speech_Chunk = np.array([], dtype=np.float32)
                        Silence_Chunk = np.array([], dtype=np.float32)

# Translate Speech to Text


def speech_to_text(IN_Speech_Q, Text_Q, OUT_Text_Q):
    Processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    Whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium")#.to('cuda')
    Forced_decoder_ids = Processor.get_decoder_prompt_ids(
        language="arabic", task="translate")
    while True:
        if not IN_Speech_Q.empty():
            input_speech = np.array(IN_Speech_Q.get(), dtype=np.float32)

            # Start MLflow run for speech-to-text translation
            with mlflow.start_run(run_name="speech_to_text") as run:
                mlflow.log_param("model", "whisper-medium")
                mlflow.log_param("task", "translate")

                input_features = Processor(
                    input_speech, sampling_rate=16000, return_tensors="pt").input_features#.to('cuda')

                # Generate token ids
                predicted_ids = Whisper_model.generate(
                    input_features, forced_decoder_ids=Forced_decoder_ids)

                # Decode token ids to text
                transcription = Processor.batch_decode(
                    predicted_ids, skip_special_tokens=True)

                Text_Q.put(transcription)
                OUT_Text_Q.put(transcription[0])
                print(transcription)

                # Log transcription as an artifact
                mlflow.log_text(transcription[0], "transcription.txt")

# Convert Text to Speech


def text_to_speech_auto(Text_Q, OUT_Speech_Q, Loading_Q):
    loaded = True

    while True:
        if not Text_Q.empty():
            text = Text_Q.get()[0]
            print(text)

            # Start MLflow run for text-to-speech conversion
            with mlflow.start_run(run_name="text_to_speech") as run:
                language = detect(text)
                mlflow.log_param("tts_engine", "gTTS")
                mlflow.log_param("language_detection", language)

                print(f"Detected Language: {language}")
                tts = gTTS(text=text, lang=language, slow=False)

                with io.BytesIO() as audio_buffer:
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                    audio_segment = AudioSegment.from_file(
                        audio_buffer, format="mp3")
                    waveform = np.array(audio_segment.get_array_of_samples())
                    OUT_Speech_Q.put(waveform)

                if loaded:
                    Loading_Q.put("loaded")
                    loaded = False

                # Log speech duration
                # Example calculation for speech duration
                duration = len(waveform) / 16000
                mlflow.log_metric("speech_duration_seconds", duration)


# Starting Threads
if __name__ == "__main__":

    if 'loading_state' not in st.session_state:
        st.session_state.loading_state = False

    if 'Loading_Q' not in st.session_state:
        st.session_state.Loading_Q = mp.Queue()

    if 'Denum_Q' not in st.session_state:
        st.session_state.Denum_Q = mp.Queue()

    if 'Receiving_Q' not in st.session_state:
        st.session_state.Receiving_Q = mp.Queue()

    if 'IN_Speech_Q' not in st.session_state:
        st.session_state.IN_Speech_Q = mp.Queue()
        _, d = wavfile.read(r"ready.wav")
        d = np.array(d / 32767, dtype=np.float32)
        st.session_state.IN_Speech_Q.put(d)

    if 'Text_Q' not in st.session_state:
        st.session_state.Text_Q = mp.Queue()

    if 'Comm_Q' not in st.session_state:
        st.session_state.Comm_Q = mp.Queue()

    if 'OUT_Speech_Q' not in st.session_state:
        st.session_state.OUT_Speech_Q = mp.Queue()

    if 'OUT_Text_Q' not in st.session_state:
        st.session_state.OUT_Text_Q = mp.Queue()

    if 'p1' not in st.session_state:
        st.session_state.p1 = mp.Process(target=Record_thread, args=(
            st.session_state.Receiving_Q, st.session_state.Comm_Q))
        st.session_state.p1.start()

    if 'p2' not in st.session_state:
        st.session_state.p2 = mp.Process(target=VAD_thread, args=(
            st.session_state.Receiving_Q, st.session_state.IN_Speech_Q, st.session_state.Denum_Q))
        st.session_state.p2.start()

    if 'p3' not in st.session_state:
        st.session_state.p3 = mp.Process(target=speech_to_text, args=(
            st.session_state.IN_Speech_Q, st.session_state.Text_Q, st.session_state.OUT_Text_Q))
        st.session_state.p3.start()

    if 'p4' not in st.session_state:
        st.session_state.p4 = mp.Process(target=text_to_speech_auto, args=(
            st.session_state.Text_Q, st.session_state.OUT_Speech_Q,  st.session_state.Loading_Q))
        st.session_state.p4.start()

    if not st.session_state.loading_state:
        with st.spinner("Wait while loading models..."):
            while True:
                if st.session_state.Loading_Q.get() == "loaded":
                    st.session_state.loading_state = True
                    break

    with st.sidebar:
        st.markdown(
            """<font color="black" face="Arial Narrow" size="5"><b>⚙️ Adjust Preferences</b></font>""", unsafe_allow_html=True)
        Main_Form = st.form("main_form", border=False)
        Output_Speech = Main_Form.radio(
            "**🎧 Listen To:**", ("Translated Speech", "Original Speech"))
        Main_Form.markdown('---')
        Showing_Transcript = Main_Form.checkbox(
            "**💬 Show Transcript !**", value=False)
        Start_butt = Main_Form.form_submit_button(
            "Apply / Start", use_container_width=True)
        Stop_butt = st.button("Stop", use_container_width=True)

    st.markdown("""<center><font color="black" face="Hacen Casablanca" size="7"><b>CONFERENCES</b></font></center>""", unsafe_allow_html=True)
    st.markdown("""<center><font color="black" face="Aktiv Grotesk" size="5"><b>🎙️ Speech Real-Time Translation</b></font></center>""", unsafe_allow_html=True)

    Messages_Container = st.container()

    OUT_Text = st.empty()

    if Start_butt:
        st.session_state.Denum_Q.put("32767")
        Messages_Container.write("🗣️:  :green[Start talking now...]")
        if Output_Speech == "Original Speech":
            st.session_state.Comm_Q.put("dev_org")
        else:
            st.session_state.Comm_Q.put("dev")

    if Stop_butt:
        st.session_state.Comm_Q.put("stop")

    while True:
        while st.session_state.OUT_Speech_Q.empty():
            pass
        out_speech = st.session_state.OUT_Speech_Q.get()
        out_text = st.session_state.OUT_Text_Q.get()
        if Showing_Transcript:
            OUT_Text.markdown(out_text)
        if Output_Speech == "Translated Speech":
            sd.play(out_speech, 25000)
            sd.wait()
