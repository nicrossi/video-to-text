import os
import sys
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import argparse
import threading
import time
from typing import Optional
from contextlib import contextmanager


MODEL_NAME = "gemini-pro-latest"
TEMP_AUDIO_FILENAME = "temp_audio_for_transcription.mp3"
AUDIO_CODEC = "mp3"
AUDIO_MIME_TYPE = "audio/mp3"
TRANSCRIPTION_FILE_SUFFIX = "_gemini_transcription.txt"
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
TRANSCRIPTION_PROMPT = "Transcribe this audio file. Provide the complete transcription."
OUTPUT_DIR = "output"


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output temporarily at the file descriptor level."""
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    # Open /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        sys.stderr.flush()
        # Redirect stderr to /dev/null
        os.dup2(devnull_fd, original_stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


class ProgressSpinner:
    """
    Spinner to show progress during long-running operations.
    """
    def __init__(self, message: str = "") -> None:
        self.message = message
        self.spinning = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.current_index = 0

    def start(self) -> None:
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self.spinner_thread.start()

    def stop(self) -> None:
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def _spin(self) -> None:
        while self.spinning:
            spinner_char = self.spinner_chars[self.current_index]
            sys.stdout.write(f'\r{spinner_char} {self.message}')
            sys.stdout.flush()
            self.current_index = (self.current_index + 1) % len(self.spinner_chars)
            time.sleep(0.1)


def transcribe_video_with_gemini(video_file_path: str) -> None:
    """
    Transcribes a video file to text using Google's Gemini AI model.
    """
    api_key = _get_api_key()
    if not api_key:
        return

    print(f"Loading video from: {video_file_path}")
    temp_audio_file_path = TEMP_AUDIO_FILENAME
    audio_extraction_successful = _extract_audio_from_video(
        video_file_path,
        temp_audio_file_path
    )
    if not audio_extraction_successful:
        return

    spinner: Optional[ProgressSpinner] = None
    try:
        spinner = ProgressSpinner("Transcribing audio: ")
        spinner.start()

        transcription_text = _transcribe_audio_with_gemini(temp_audio_file_path)

        output_transcription_file_path = _generate_output_filename(video_file_path)
        _save_transcription_to_file(transcription_text, output_transcription_file_path)

        # print("\n==== Transcription ====")
        # print(transcription_text)

    except Exception as transcription_error:
        print(f"An error occurred during transcription: {transcription_error}")

    finally:
        if spinner:
            spinner.stop()

        _cleanup_temporary_audio_file(temp_audio_file_path)


def _get_api_key() -> Optional[str]:
    """
    Retrieves and validates the API key from environment variables.
    """
    try:
        api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            print(f"Error: {API_KEY_ENV_VAR} environment variable not set.")
            print(f"Please set your API key using: export {API_KEY_ENV_VAR}='your-api-key'")
            return None

        with suppress_stderr():
            genai.configure(api_key=api_key)

        return api_key

    except Exception as configuration_error:
        print(f"Error configuring the Gemini API: {configuration_error}")
        return None


def _extract_audio_from_video(video_file_path: str, output_audio_path: str) -> bool:
    try:
        print("Extracting audio from video...")
        with VideoFileClip(video_file_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(output_audio_path, codec=AUDIO_CODEC)

        print(f"Audio extracted successfully to: {output_audio_path}")
        return True

    except Exception as extraction_error:
        print(f"Error during audio extraction: {extraction_error}")
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
            print(f"Cleaned up temporary audio file: {output_audio_path}")
        return False


def _transcribe_audio_with_gemini(audio_file_path: str) -> str:
    """
    Sends audio file to Gemini API for transcription.
    """
    gemini_model = genai.GenerativeModel(model_name=MODEL_NAME)
    with open(audio_file_path, 'rb') as audio_file:
        audio_binary_data = audio_file.read()

    print("Sending audio to Gemini for transcription...")
    transcription_prompt =  TRANSCRIPTION_PROMPT
    # Send audio data with inline binary content
    # suppress gRPC and ALTS warnings
    with suppress_stderr():
        gemini_response = gemini_model.generate_content([
            transcription_prompt,
            {
                "mime_type": AUDIO_MIME_TYPE,
                "data": audio_binary_data
            }
        ])

    transcription_text = gemini_response.text
    print("Transcription received from Gemini successfully.")

    return transcription_text


def _generate_output_filename(video_file_path: str) -> str:
    video_filename_without_extension = os.path.splitext(os.path.basename(video_file_path))[0]
    output_filename = f"{video_filename_without_extension}{TRANSCRIPTION_FILE_SUFFIX}"
    return os.path.join(OUTPUT_DIR, output_filename)

def _save_transcription_to_file(transcription_text: str, output_file_path: str) -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Saving transcription to: {output_file_path}")
    with open(output_file_path, "w", encoding='utf-8') as output_file:
        output_file.write(transcription_text)
    print(f"Transcription saved successfully to: {output_file_path}")


def _cleanup_temporary_audio_file(audio_file_path: str) -> None:
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
        print(f"\nTemporary audio file '{audio_file_path}' has been deleted.")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Transcribe an MP4 video file to text using Google's Gemini AI model."
    )
    argument_parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to transcribe."
    )
    parsed_arguments = argument_parser.parse_args()

    video_file_path = parsed_arguments.video_path
    if not os.path.exists(video_file_path):
        print(f"Error: The file '{video_file_path}' does not exist.")
        print("Please provide a valid path to a video file.")
        return

    transcribe_video_with_gemini(video_file_path)


if __name__ == "__main__":
    main()
