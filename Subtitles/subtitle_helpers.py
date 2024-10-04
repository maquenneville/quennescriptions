# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 00:27:39 2023

@author: marca
"""

import os
import subprocess
import openai
from openai import OpenAI
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
import glob
import traceback
import srt
from datetime import timedelta
import configparser

def get_api_keys(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
    return openai_api_key

# Set up the OpenAI API client
OPENAI_API_KEY = get_api_keys('config.ini')
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_audio(video_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    print("Starting audio extraction...")
    audio_filename = os.path.splitext(video_path)[0] + ".wav"
    try:
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            video_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_filename,
            "-y",  # Overwrite output files without asking
        ]
        subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print(f"Audio extracted and saved to {audio_filename}")
        return audio_filename
    except FileNotFoundError:
        print("FFmpeg not found. Please ensure FFmpeg is installed and the path is correct.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio: {e}")
        return None

def transcribe_using_whisper(audio_file):
    """
    Transcribes audio using OpenAI's Whisper API and returns JSON response.
    """
    print("Starting transcription using Whisper API...")
    try:
        with open(audio_file, "rb") as af:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                response_format="verbose_json"
            )
        print("Transcription complete!")
        return transcript  # This is the JSON response
    except openai.error.OpenAIError as e:
        print(f"An error occurred during transcription: {e}")
        return None

def generate_srt_from_transcript(transcript_json, srt_filename):
    """
    Generates an SRT file from the transcript JSON data.
    """
    print("Generating SRT file from transcript...")
    segments = transcript_json.segments
    subtitles = []
    for i, segment in enumerate(segments):
        index = i + 1
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        content = segment['text'].strip()
        subtitle = srt.Subtitle(index, start_time, end_time, content)
        subtitles.append(subtitle)
    srt_content = srt.compose(subtitles)
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)
    print(f"SRT file saved to {srt_filename}")


def add_subtitles_to_video(video_path, srt_path):
    """
    Adds subtitles to a video file using an SRT file and saves the output.

    Args:
        video_path (str): The path to the original video file.
        srt_path (str): The path to the SRT subtitle file.

    Returns:
        str: The path to the subtitled video file.
    """
    print("Adding subtitles to video...")
    video_clip = VideoFileClip(video_path)
    video_width = video_clip.w  # Video width
    video_height = video_clip.h  # Video height

    # Define the generator function
    def subtitle_generator(txt):
        return TextClip(
            txt,
            font="Arial",
            fontsize=int(video_height * 0.05),  # Maintain current fontsize
            color="white",
            bg_color="black",  # Add black background
            method='caption',  # Use 'caption' method for text wrapping
            size=(int(video_width * 0.8), None),  # Set width to 80% of video width
            align='center',  # Center align text
        )



    subtitles_clip = SubtitlesClip(srt_path, subtitle_generator)

    result = CompositeVideoClip(
        [video_clip, subtitles_clip.set_pos(('center', 'bottom'))]
    )

    output_filename = os.path.splitext(video_path)[0] + "_subtitled.mp4"

    result.write_videofile(
        output_filename,
        fps=video_clip.fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )
    print(f"Subtitled video saved to {output_filename}")
    return output_filename


def cleanup_temp_files(video_path):
    """
    Deletes temporary files created by the program.
    """
    print("Performing cleanup of temporary files...")
    base_name = os.path.splitext(video_path)[0]

    # List of patterns to match temporary files
    temp_files_patterns = [
        f"{base_name}.wav",
        f"{base_name}.srt",
        "temp_chunk_*.wav",
        "temp-audio.m4a",
    ]

    for pattern in temp_files_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted temporary file: {file}")
            except Exception as e:
                print(f"Could not delete file {file}: {e}")

def generate_subtitles(video_path):
    """
    Full pipeline to generate subtitles for a video using Whisper.

    Args:
        video_path (str): The path to the video file.
    """
    print("Generating subtitles...")
    # Extract audio from video
    audio_path = extract_audio(video_path)
    if not audio_path:
        return

    # Transcribe audio using Whisper and get JSON response
    transcript_json = transcribe_using_whisper(audio_path)
    if not transcript_json:
        return

    # Generate SRT file from JSON transcript
    srt_filename = os.path.splitext(video_path)[0] + ".srt"
    generate_srt_from_transcript(transcript_json, srt_filename)

    # Add subtitles to video using the SRT file
    add_subtitles_to_video(video_path, srt_filename)

    print("Subtitle generation completed successfully.")
                                                                                                                                                                                                                                                                                                                                                                                