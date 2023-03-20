# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 00:27:39 2023

@author: marca
"""

import os
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import io
import numpy as np
import webrtcvad
from pydub import AudioSegment
import speech_recognition as sr
import pyaudio
import wave
import cv2
import subprocess
import noisereduce as nr
from pydub.silence import detect_nonsilent
import librosa
import openai
from openai.error import RateLimitError
import time
import math

# Set up the OpenAI API client
openai.api_key = "YOUR_API_KEY"


def compress_wav_to_flac(wav_filename):
    """
    Compresses a WAV file to FLAC format using the flac command-line tool.

    Args:
    wav_filename (str): the name of the WAV file to compress

    Returns:
    The name of the resulting FLAC file
    """
    # Create FLAC filename
    flac_filename = os.path.splitext(wav_filename)[0] + ".flac"

    # Check if a file with the same name already exists
    if os.path.exists(flac_filename):
        os.remove(flac_filename)

    # Compress WAV file to FLAC format using flac command-line tool
    subprocess.run(["flac", "--silent", "-o", flac_filename, wav_filename], check=True)

    print("WAV file converted to FLAC")
    print(flac_filename)
    return flac_filename


def convert_videos_to_wav(folder_path):
    for file in os.listdir(folder_path):
        # Check if the file is a video file
        if file.endswith((".mp4", ".mov", ".avi", ".mkv")):
            # Create the input and output filenames for the MP3 and WAV files
            extension = os.path.splitext(file)[1][1:]
            input_file = os.path.join(folder_path, file)
            mp3_file = os.path.splitext(input_file)[0] + extension + ".mp3"
            wav_file = os.path.splitext(input_file)[0] + extension + ".wav"

            try:
                subprocess.run(["ffmpeg", "-i", input_file, mp3_file], check=True)
                subprocess.run(["ffmpeg", "-i", mp3_file, wav_file], check=True)
                os.remove(mp3_file)
                print(f"Successfully converted {file} to {wav_file}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {file}: {e}")


def convert_single_to_wav(video_path):

    mp3_file = os.path.splitext(video_path)[0] + ".mp3"
    wav_file = os.path.splitext(video_path)[0] + ".wav"

    try:
        subprocess.run(["ffmpeg", "-i", video_path, mp3_file], check=True)
        subprocess.run(["ffmpeg", "-i", mp3_file, wav_file], check=True)
        os.remove(mp3_file)
        print("Successfully converted file to wav_file")
        return wav_file
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert file: {e}")


def extract_audio(video_path, audio_filename=None):
    # Load the video file
    video = mp.VideoFileClip(video_path)

    # Extract the audio from the video file
    audio = video.audio

    # Create a filename for the audio file in the same folder as the video file
    if not audio_filename:
        audio_filename = os.path.splitext(video_path)[0] + ".wav"
    else:
        audio_filename = audio_filename + ".wav"

    # Save the audio to the WAV file
    audio.write_audiofile(
        audio_filename, codec="pcm_s16le", fps=44100, nbytes=2, bitrate="16k"
    )

    print("Audio extracted and saved to", audio_filename)


def transcribe_speech_audio(raw_audio_file):
    # Load the raw PCM data from a file
    audio_data = np.fromfile(raw_audio_file, dtype=np.int16)

    # Create an AudioData object from the raw PCM data
    audio = sr.AudioData(audio_data.tobytes(), 16000, 1)

    # Set up the SpeechRecognition recognizer
    recognizer = sr.Recognizer()

    # Perform the transcription using the Google Web Speech API
    try:
        transcribed_text = recognizer.recognize_google(audio)
        return transcribed_text
    except sr.UnknownValueError:
        print("Unable to transcribe audio")


def transcribe_audio(audio_file, dialect_code="en-US"):
    # Load the audio file
    with sr.WavFile(audio_file) as source:
        # Adjust for ambient noise
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.0

        recognizer.adjust_for_ambient_noise(source)

        # Extract audio data from the file
        audio_data = recognizer.record(source)

    # Transcribe the audio data using the Google Web Speech API
    try:
        transcribed_text_google = recognizer.recognize_google(audio_data)
        return transcribed_text_google
    except sr.UnknownValueError:
        print("Unable to transcribe audio")


def record_audio(filename, duration=5, sample_rate=44100, channels=1, sample_width=2):
    """
    Records audio from the default microphone and saves it to a .wav file.

    :param filename: The name of the output .wav file.
    :param duration: The duration of the recording in seconds (default is 5 seconds).
    :param sample_rate: The sample rate of the audio (default is 44100 Hz).
    :param channels: The number of channels (default is 1).
    :param sample_width: The sample width in bytes (default is 2 bytes).
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the microphone stream
    stream = p.open(
        format=p.get_format_from_width(sample_width),
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Recording...")

    # Record audio data from the microphone
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording done.")

    # Stop the microphone stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio data as a .wav file
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"Saved audio to {filename}.")


def record_video(video_path, length_seconds):
    # Set up the video capture device
    cap = cv2.VideoCapture(0)

    # Set the video resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Set the video codec and output file extension for each file type
    codec = cv2.VideoWriter_fourcc(*"XVID")
    avi_ext = ".avi"
    mp4_ext = ".mp4"
    mkv_ext = ".mkv"

    # Calculate the total number of frames to record
    total_frames = int(length_seconds * cap.get(cv2.CAP_PROP_FPS))

    # Set up the video writers for each file type
    avi_path = os.path.splitext(video_path)[0] + avi_ext
    avi_writer = cv2.VideoWriter(avi_path, codec, cap.get(cv2.CAP_PROP_FPS), (640, 480))

    mp4_path = os.path.splitext(video_path)[0] + mp4_ext
    mp4_writer = cv2.VideoWriter(
        mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (640, 480)
    )

    mkv_path = os.path.splitext(video_path)[0] + mkv_ext
    mkv_writer = cv2.VideoWriter(
        mkv_path, cv2.VideoWriter_fourcc(*"VP90"), cap.get(cv2.CAP_PROP_FPS), (640, 480)
    )

    # Start recording video
    print("Recording...")
    for i in range(total_frames):
        ret, frame = cap.read()

        if ret:
            avi_writer.write(frame)
            mp4_writer.write(frame)
            mkv_writer.write(frame)
        else:
            print("Error capturing frame")

    # Release the video capture device and video writers
    cap.release()
    avi_writer.release()
    mp4_writer.release()
    mkv_writer.release()

    print("Video saved to:", avi_path, mp4_path, mkv_path)


def convert_mp4_to_avi(mp4_path):
    # Load the MP4 file as a VideoFileClip object
    clip = mp.VideoFileClip(mp4_path)

    # Set the output AVI file path and extension
    avi_path = os.path.splitext(mp4_path)[0] + ".avi"

    # Write the AVI file using the same codec and parameters as the input
    clip.write_videofile(avi_path, codec="png", fps=clip.fps)

    # Close the VideoFileClip object to free up memory
    clip.close()

    print("mp4 file converted to avi")


def convert_mp4_to_mkv(mp4_path):
    # Load the MP4 file as a VideoFileClip object
    clip = mp.VideoFileClip(mp4_path)

    # Set the output AVI file path and extension
    avi_path = os.path.splitext(mp4_path)[0] + ".mkv"

    # Write the AVI file using the same codec and parameters as the input
    clip.write_videofile(avi_path, codec="png", fps=clip.fps)

    # Close the VideoFileClip object to free up memory
    clip.close()

    print("mp4 file converted to mkv")


def enhance_background(input_file):
    # Load the input WAV file
    audio = AudioSegment.from_file(input_file)

    # Convert to a numpy array for processing
    samples = np.array(audio.get_array_of_samples())

    # Split channels and reduce noise on each channel separately
    channels = librosa.core.audio.channels_to_samples(samples, audio.channels)
    for c in range(audio.channels):
        channels[c] = nr.reduce_noise(
            audio_clip=channels[c], noise_clip=channels[c], verbose=False
        )

    # Recombine channels
    new_samples = np.hstack(channels)

    # Convert back to AudioSegment and save
    new_audio = AudioSegment(
        new_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels,
    )

    # Adjust audio to improve readability for transcription
    adjusted_audio = librosa.effects.trim(new_audio.get_array_of_samples(), top_db=20)[
        0
    ]
    adjusted_audio = librosa.effects.preemphasis(adjusted_audio)

    # Create a filename for the enhanced audio file in the same folder as the input file
    output_file = os.path.splitext(input_file)[0] + "_background.wav"

    # Save the enhanced audio to a new WAV file while preserving the metadata
    new_audio.export(output_file, format="wav")

    print("Background audio extracted and saved to", output_file)
    return output_file


def generate_edited_response(prompt, dialogue=False):

    if dialogue:

        messages = [
            {
                "role": "system",
                "content": "This chat is for formatting and correcting raw text into properly formatted dialogue.",
            },
            {
                "role": "user",
                "content": f"Format the following text into dialogue as best you can: '{prompt}'",
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "This chat is for formatting and punctuating raw text into properly formatted sentences, with each sentence on a new line.",
            },
            {
                "role": "user",
                "content": f"Format the following text.  Do not add or switch words: '{prompt}'",
            },
        ]

    model_engine = "gpt-3.5-turbo"
    # Generate a response
    max_retries = 10
    retries = 0
    while True:
        if retries < max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_engine,
                    messages=messages,
                    n=1,
                    temperature=0.5,
                )
                break
            except (RateLimitError, KeyboardInterrupt):
                time.sleep(60)
                retries += 1
                print("Server overloaded, retrying in a minute")
                continue
        else:
            print("Failed to generate prompt after max retries")
            return
    response = completion.choices[0].message.content
    return response


def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-1):
    n, m = len(seq1), len(seq2)
    dp_matrix = np.zeros((n + 1, m + 1))
    traceback = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        dp_matrix[i, 0] = i * gap_penalty
        traceback[i, 0] = 1

    for j in range(1, m + 1):
        dp_matrix[0, j] = j * gap_penalty
        traceback[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp_matrix[i - 1, j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score
            )
            delete = dp_matrix[i - 1, j] + gap_penalty
            insert = dp_matrix[i, j - 1] + gap_penalty
            dp_matrix[i, j] = max(match, delete, insert)

            if dp_matrix[i, j] == match:
                traceback[i, j] = 0
            elif dp_matrix[i, j] == delete:
                traceback[i, j] = 1
            else:
                traceback[i, j] = 2

    i, j = n, m
    aligned_seq1, aligned_seq2 = [], []
    while i > 0 or j > 0:
        if traceback[i, j] == 0:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()
    return aligned_seq1, aligned_seq2



def find_speech_segments(audio_path, vad_level=3, frame_duration_ms=30, min_speech_gap=1200, min_duration=2.0):
    audio = AudioSegment.from_wav(audio_path).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    vad = webrtcvad.Vad(vad_level)
    num_frames = len(audio) // frame_duration_ms
    audio_frames = [audio[i * frame_duration_ms:(i + 1) * frame_duration_ms] for i in range(num_frames)]

    speech_segments = []
    current_segment = None
    speech_buffer = []

    for i, frame in enumerate(audio_frames):
        frame_start = i * frame_duration_ms / 1000
        frame_end = frame_start + frame_duration_ms / 1000
        is_speech = vad.is_speech(frame.raw_data, sample_rate=16000)

        if is_speech:
            speech_buffer.append((frame_start, frame_end))
            current_segment = (current_segment[0] if current_segment else frame_start, frame_end)
        elif current_segment and len(speech_buffer) * frame_duration_ms >= min_speech_gap:
            current_segment = (current_segment[0], current_segment[1] + max(0, min_duration - (current_segment[1] - current_segment[0])))
            speech_segments.append(current_segment)
            current_segment = None
            speech_buffer = []

    if current_segment:
        current_segment = (current_segment[0], current_segment[1] + max(0, min_duration - (current_segment[1] - current_segment[0])))
        speech_segments.append(current_segment)

    return speech_segments



def create_subtitle_tuples(speech_segments, transcription):
    lines = transcription.split('\n')
    num_segments = min(len(speech_segments), len(lines))

    subtitle_tuples = [((start, end), line) for (start, end), line in zip(speech_segments[:num_segments], lines[:num_segments])]

    return subtitle_tuples



def create_subtitles(audio_path, transcription, min_speech_gap=900):
    
    print("Scanning audio for speech segments...")
    # Find the speech segments
    speech_segments = find_speech_segments(audio_path, min_speech_gap)
    print("Speech segments found, creating subtitles...")
    # Create a list of tuples representing the subtitles
    subtitles = create_subtitle_tuples(speech_segments, transcription)
    
    print("Subtitles created!")
    return subtitles


def add_subtitles_to_video(video_path, subtitles):
    # Create a text clip generator
    generator = lambda txt: TextClip(txt, font="Arial", fontsize=24, color="white")

    # Create the subtitles clip
    subtitles_clip = SubtitlesClip(subtitles, generator)

    # Load the input video clip
    video_clip = VideoFileClip(video_path)

    # Combine the video and subtitles clips
    result = CompositeVideoClip(
        [video_clip, subtitles_clip.set_pos(("center", "bottom"))]
    )

    # Write the output video file
    output_filename = os.path.splitext(video_path)[0] + "_subtitled.mp4"
    result.write_videofile(
        output_filename,
        fps=video_clip.fps,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        codec="libx264",
        audio_codec="aac",
    )
