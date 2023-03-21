# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 00:35:16 2023

@author: marca
"""


import argparse
import os
from VideoSubtitlesHelpers import (
    convert_single_to_wav,
    transcribe_using_whisper,
    generate_edited_response,
    create_subtitles,
    add_subtitles_to_video,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert video to WAV, transcribe speech, generate edited response, and add transcript to video."
    )
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument(
        "--speech_gap",
        help="Max gap between words of the same phrase in milliseconds (default is 900 ms)",
        default=900,
    )
    parser.add_argument(
        "--dialogue",
        help="Set this flag if the transcription is dialogue",
        action="store_true",
    )
    args = parser.parse_args()

    # Convert the video to WAV format
    print("Converting video to WAV format...")
    audio_file = convert_single_to_wav(args.video_file)

    # Transcribe the speech in the audio file
    print("Transcribing speech...")
    transcript = transcribe_using_whisper(audio_file)

    # Generate the edited response to the prompt
    print("Generating edited response...")
    final_transcript = generate_edited_response(transcript, dialogue=args.dialogue)

    # Add the transcript to the video
    print("Adding transcript to video...")
    subs = create_subtitles(audio_file, final_transcript, min_speech_gap=args.speech_gap)
    add_subtitles_to_video(args.video_file, subs)

    # Clean up intermediate files
    os.remove(audio_file)

    print("Done!")


if __name__ == "__main__":
    main()
