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
from typing import Optional
import re
import json

def get_api_keys(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
    return openai_api_key

# Set up the OpenAI API client
OPENAI_API_KEY = get_api_keys('config.ini')
client = OpenAI(api_key=OPENAI_API_KEY)



def convert_to_mp4(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Converts a video file to MP4 format using FFmpeg with lossless settings.
    
    Args:
        input_path (str): The file path of the input video.
        output_path (str, optional): The desired file path for the output MP4 video.
                                      If not provided, the function will generate an output path
                                      by replacing the input file's extension with '.mp4'.
    
    Returns:
        Optional[str]: The file path of the converted MP4 video if successful, else None.
    """
    # Check if the input file exists
    if not os.path.isfile(input_path):
        print(f"Input file does not exist: {input_path}")
        return None

    # If output_path is not provided, replace the input file's extension with .mp4
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.mp4"

    # If the input file is already an MP4, skip conversion
    if input_path.lower().endswith('.mp4'):
        print(f"The file is already in MP4 format: {input_path}")
        return input_path

    # Construct the FFmpeg command for lossless video encoding
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_path,          # Input file
        "-c:v", "libx264",         # Video codec: H.264
        "-preset", "veryslow",     # Preset: veryslow for better compression
        "-crf", "0",               # CRF: 0 for lossless
        "-c:a", "copy",            # Attempt to copy audio stream
        "-y",                      # Overwrite output files without asking
        output_path                # Output file
    ]

    print(f"Converting '{input_path}' to MP4 format with lossless settings...")

    try:
        # Execute the FFmpeg command
        result = subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print(f"Conversion successful! MP4 file saved at: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with error:\n{e.stderr.strip()}")
        # Attempt to fallback to AAC encoding with high bitrate if audio copy fails
        if "Stream mapping:" in e.stderr:
            print("Attempting to encode audio with AAC at high bitrate...")
            ffmpeg_command_audio_fallback = [
                "ffmpeg",
                "-i", input_path,          # Input file
                "-c:v", "libx264",         # Video codec: H.264
                "-preset", "veryslow",     # Preset: veryslow for better compression
                "-crf", "0",               # CRF: 0 for lossless
                "-c:a", "aac",             # Audio codec: AAC
                "-b:a", "320k",            # Audio bitrate: 320 kbps
                "-y",                      # Overwrite output files without asking
                output_path                # Output file
            ]
            try:
                subprocess.run(
                    ffmpeg_command_audio_fallback,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                print(f"Conversion successful with AAC audio! MP4 file saved at: {output_path}")
                return output_path
            except subprocess.CalledProcessError as e2:
                print(f"FFmpeg failed again with error:\n{e2.stderr.strip()}")
                return None
        else:
            return None
    except FileNotFoundError:
        print("FFmpeg not found. Please ensure FFmpeg is installed and the path is correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        return None


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
                response_format="verbose_json",
                timestamp_granularities=['word', 'segment']
            )
        print("Transcription complete!")
        #print(transcript)
        return transcript  # This is the JSON response
    except openai.error.OpenAIError as e:
        print(f"An error occurred during transcription: {e}")
        return None

def generate_srt_from_transcript(transcript_json, srt_filename):
    """
    Generates an SRT file from the transcript JSON data with incremental word additions within segments.
    """
    print("Generating SRT file from transcript...")
    words = transcript_json.words
    segments = transcript_json.segments
    subtitles = []
    index = 1  # Subtitle index
    
    word_idx = 0  # Index to keep track of the current word
    
    for segment in segments:
        current_text = ""
        segment_words = []
        
        # Collect words that belong to this segment
        while word_idx < len(words) and words[word_idx]['start'] >= segment['start'] and words[word_idx]['end'] <= segment['end']:
            segment_words.append(words[word_idx])
            word_idx += 1

        num_words = len(segment_words)
        for i, word in enumerate(segment_words):
            current_text += word['word'] + ' '
            start_time = word['start']
            if i + 1 < num_words:
                end_time = segment_words[i + 1]['start']
            else:
                end_time = segment['end']
            subtitle = srt.Subtitle(
                index=index,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=current_text.strip()
            )
            subtitles.append(subtitle)
            index += 1

    srt_content = srt.compose(subtitles)
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)
    print(f"SRT file saved to {srt_filename}")




def add_subtitles_to_video(video_path, srt_path, is_portrait=False, font_size=None):
    """
    Adds subtitles to a video file using an SRT file and saves the output.

    Args:
        video_path (str): The path to the original video file.
        srt_path (str): The path to the SRT subtitle file.
        is_portrait (bool): Indicates if the video is in portrait mode.
        font_size (int, optional): The font size for the subtitles.
    """
    print("Adding subtitles to video...")
    video_clip = VideoFileClip(video_path)

    # Swap width and height if in portrait mode
    if is_portrait:
        # Swap the width and height to correct the aspect ratio
        new_width, new_height = video_clip.h, video_clip.w
        video_clip = video_clip.resize((new_width, new_height))
    else:
        new_width, new_height = video_clip.w, video_clip.h

    # Define the generator function
    def subtitle_generator(txt):
        fontsize = font_size if font_size else int(new_height * 0.05)
        return TextClip(
            txt,
            font="Arial",
            fontsize=fontsize,
            color="white",
            bg_color="black",
            method='caption',
            size=(int(new_width * 0.8), None),
            align='center',
        )

    subtitles_clip = SubtitlesClip(srt_path, subtitle_generator)

    result = CompositeVideoClip(
        [video_clip, subtitles_clip.set_pos(('center', 'bottom'))],
        size=(new_width, new_height)  # Explicitly set the size here
    )

    output_filename = os.path.splitext(video_path)[0] + "_subtitled.mp4"

    result.write_videofile(
        output_filename,
        fps=video_clip.fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=8
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



def generate_subtitles(video_path, is_portrait=False, font_size=None):
    """
    Full pipeline to generate subtitles for a video using Whisper.

    Args:
        video_path (str): The path to the video file.
        is_portrait (bool): Indicates if the video is in portrait mode.
        font_size (int, optional): The font size for the subtitles.
    """
    print("Generating subtitles...")

    # Convert video to MP4 if necessary
    converted_video_path = convert_to_mp4(video_path)
    if not converted_video_path:
        return
    else:
        video_path = converted_video_path

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
    add_subtitles_to_video(video_path, srt_filename, is_portrait, font_size)

    print("Subtitle generation completed successfully.")










