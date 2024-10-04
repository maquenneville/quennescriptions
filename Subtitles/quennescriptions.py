# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 00:35:16 2023

@author: marca
"""


import argparse
import os
from subtitle_helpers import generate_subtitles


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate subtitles for a video using OpenAI's Whisper API."
    )
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads to use for transcription (default: 4)",
    )
    parser.add_argument(
        "--min_silence_length",
        type=int,
        default=500,
        help="Minimum length of silence to consider a break in milliseconds (default: 500 ms)",
    )
    parser.add_argument(
        "--silence_thresh",
        type=int,
        default=-40,
        help="Silence threshold in dBFS (default: -40 dBFS)",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=24,
        help="Font size for subtitles (default: 24)",
    )
    args = parser.parse_args()

    video_file = args.video_file

    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
        return

    # Generate subtitles
    generate_subtitles(
        video_file,
        num_threads=args.num_threads,
        min_silence_length=args.min_silence_length,
        silence_thresh=args.silence_thresh,
        font_size=args.font_size,
    )

    print("Done!")


if __name__ == "__main__":
    main()

