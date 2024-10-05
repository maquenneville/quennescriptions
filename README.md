# Quennescriptions


This Python script provides a command-line interface for adding speech-to-text transcription for video files, creating a copy with subtitles (accepts most mainstream video filetypes).

## Installation
- to use this script, you will need to have >= Python 3.8.

- activate your environment, and install these packages using pip:

`pip install -r requirements.txt`


- update the config file with your OpenAI API key (information located here: https://platform.openai.com/overview)


## Usage
To use the script, first change the API key in VideoSubtitlesHelpers.py to your OpenAI API key.  Then, run the main.py file from the command line with the following arguments:

`python quennescriptions.py <video_file> [--portrait] [--font-size FONT SIZE]`

(replace <video_file> with the path to your video file.  If the video is in portrait orientation, try adding the portait flag (defaults to landscape).  Add a font size to tweak the default, dynamically-generated font size).

The script will then:

- Extract the audio from the provided video file and saves it as a WAV file.
Transcribe the Speech:

- Use OpenAI's Whisper API to transcribe the audio from the WAV file.

- Format the raw transcription into an SRT (SubRip Subtitle) file, ensuring accurate timing and formatting.

- Embed the generated subtitles into a copy of the original video, producing a new video file with subtitles.

- The resulting video file with the added subtitles will be saved in the same directory as the original video file (The filename will retain the original name with the suffix _subtitled.mp4)


## License
This script is licensed under the MIT License. See the LICENSE file for more information.

## Notes

You may need to set the ImageMagick config_default.py variable IMAGEMAGICK_BINARY to the path with your magick.exe file, especially if you're using Windows.


## Updates

10/4/2024 -- Leveraged Whisper's segment timing to improve subtitle placement and display timing, streamlining the process greatly.  Should see much improved performance.

3/21/2023 -- Upgraded transcription method to OpenAI Whisper API calls, for a massive improvement in subtitle accuracy.  Removed --dialect option as Whisper does not need to be prompted for dialect.

3/19/2023 -- Abandoned Needleman-Wunsch, optimized old VAD function for minimum phrase length.
  
3/14/2023 -- Major overhaul to method used to generate speech segments and match them with transcribed lines using Needleman-Wunsch algorithm, aiming to improve accuracy of subtitles.

