# Video To Text (VTT)

Convert video files to text transcriptions using Google's Gemini AI.

## Setup

1. Installing dependencies from dependencies file
```bash
  # Make sure to be inside an active venv
  pip install -r dependencies.txt
```

2. Get a Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

3. Set your API key:
```bash
  export GOOGLE_API_KEY='your-api-key-here'
```

## Usage

```bash
  python vtt.py path/to/your/video.mp4
```

Transcription will be saved in the `output` folder as `video_name_gemini_transcription.txt`

## Example

```bash
  python vtt.py meeting_recording.mp4
```

Output: `output/meeting_recording_gemini_transcription.txt`
