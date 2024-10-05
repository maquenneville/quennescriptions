# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 00:35:16 2023

@author: marca
"""


import argparse
import os
from subtitle_helpers import generate_subtitles, cleanup_temp_files
import traceback

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate subtitles for a video using OpenAI's Whisper API."
    )
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument(
        "--portrait",
        action="store_true",
        help="Specify if the video is in portrait mode",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        help="Specify the font size for the subtitles",
    )
    args = parser.parse_args()

    video_file = args.video_file
    is_portrait = args.portrait
    font_size = args.font_size

    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
        return

    # Generate subtitles
    try:
        generate_subtitles(video_file, is_portrait, font_size)

    except Exception as e:
        print(traceback.format_exc())
    finally:
        # Ensure that cleanup_temp_files is called only if video_file is defined
        try:
            cleanup_temp_files(video_file)
        except NameError:
            pass


if __name__ == "__main__":
    main()


