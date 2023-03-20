# HomeVideoSubtitles


This Python script provides a command-line interface for adding speech-to-text transcription for video files, creating a copy with subtitles (accepts most mainstream video filetypes).

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


You'll also need an OpenAi developers API key, information located here: https://platform.openai.com/overview


# Usage
To use the script, first change the API key in VideoSubtitlesHelpers.py to your OpenAI API key.  Then, run the main.py file from the command line with the following arguments:

python VideoSubtitlesMain.py <video_file> [--dialect <dialect_code>] [--speech_gap] [--dialogue]

Replace <video_file> with the video file path and <prompt> with . The --dialect, speech_gap and --dialogue arguments are optional, and default to "en-US", 900ms and False respectively (leave --dialogue out if you want a narrative transcription, add it to change the transcription format to dialogue).

The script will perform the following steps:

-Convert the video file to WAV format
-Transcribe the speech in the audio file
-Clean up and format the raw transcription
-Find speech segments and pair them with the correct sentence/phrase
-Add the transcript to a copy of the video
-Clean up intermediate files.

The resulting video file with the added transcript will be saved in the same directory as the original video file, with the same filename and the suffix "_transcript".

# License
This script is licensed under the MIT License. See the LICENSE file for more information.

# Notes

You may need to set the ImageMagick config_default.py variable IMAGEMAGICK_BINARY to the path with your magick.exe file, especially if you're using Windows.

3/19/2023 -- Abandoned Needleman-Wunsch, optimized old VAD function for minimum phrase length.
  
3/14/2023 -- Major overhaul to method used to generate speech segments and match them with transcribed lines using Needleman-Wunsch algorithm, aiming to improve accuracy of subtitles.

