# HomeVideoSubtitles


This Python script provides a command-line interface for adding speech-to-text transcription for video files (accepts most mainstream video filetypes).

# Installation
To use this script, you will need to have Python 3 and the following Python packages installed:

moviepy
SpeechRecognition
numpy
pandas
os
pydub
pyaudio
wave
cv2
noisereduce
librosa
openai



You can install these packages using pip:

pip install moviepy SpeechRecognition numpy pandas pydub pyaudio wave cv2 noisereduce librosa openai


# Usage
To use the script, run the main.py file from the command line with the following arguments:

python VideoSubtitlesMain.py <video_file> [--dialect <dialect_code>] [--line-time <line_time>] [--dialogue]

Replace <video_file> with the video file path and <prompt> with . The --dialect, --line-time and --dialogue arguments are optional, and default to "en-US", 10 seconds, and False respectively (leave --dialogue out if you want a narrative transcription, add it to change the transcription format to dialogue).

The script will perform the following steps:

-Convert the video file to WAV format
-Transcribe the speech in the audio file
-Clean up and format the raw transcription
-Add the transcript to a copy of the video
-Clean up intermediate files.

The resulting video file with the added transcript will be saved in the same directory as the original video file, with the same filename and the suffix "_transcript".

# License
This script is licensed under the MIT License. See the LICENSE file for more information.

# Notes

This script is a work in progress and not yet fully stable

