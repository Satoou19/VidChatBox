import os
import time
import json
import re
from openai import OpenAI
import google.generativeai as genai

class AudioTranscriber:
    def __init__(self, gemini_api_key=None, openai_api_key=None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

    def transcribe(self, audio_path, provider="gemini", progress_callback=None):
        """Transcribes the audio file using Gemini API or OpenAI Whisper API.
        
        Returns a list of dictionaries with 'start', 'end', and 'text'.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        # Route based on selected provider
        if provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not configured in .env file.")
            if progress_callback:
                progress_callback("uploading_to_gemini", 0)
            return self._transcribe_gemini(audio_path, progress_callback)
        elif provider == "openai":
            if not self.openai_client:
                raise ValueError("OPENAI_API_KEY is not configured in .env file.")
            if file_size_mb > 25:
                raise ValueError(
                    f"Audio file size ({file_size_mb:.2f}MB) exceeds OpenAI Whisper's 25MB limit. "
                    "For large files, please install FFmpeg locally or use Gemini API."
                )
            if progress_callback:
                progress_callback("transcribing_openai", 0)
            return self._transcribe_openai_whisper(audio_path)
        else:
            raise ValueError(f"Unknown transcription provider: {provider}")

    def _transcribe_gemini(self, audio_path, progress_callback=None):
        """Uploads audio file to Google Gemini File API and transcribes it using Gemini 1.5 Flash.
        
        Requests a structured JSON response with timestamps.
        """
        # Upload using the Files API
        if progress_callback:
            progress_callback("uploading_to_gemini", 20)
            
        print(f"Uploading {audio_path} to Gemini...")
        uploaded_file = genai.upload_file(path=audio_path)
        print(f"Uploaded successfully. File name: {uploaded_file.name}")
        
        if progress_callback:
            progress_callback("processing_at_gemini", 50)
            
        # Wait for file to process (active state)
        while uploaded_file.state.name == "PROCESSING":
            print("File is processing at Gemini...")
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise Exception("Gemini File API processing failed.")
            
        print("File is ready for transcription.")
        if progress_callback:
            progress_callback("transcribing_gemini", 75)

        # We request Gemini to return a detailed transcription with timestamp ranges in JSON format.
        # This is extremely powerful as Gemini's context window easily fits the entire VOD audio.
        prompt = (
            "Transcribe this audio file. Your output must be a valid JSON array of objects. "
            "Each object must represent a short segment (typically 5-15 seconds) and have the following fields: "
            "1. 'start' (float, start time in seconds from the beginning) "
            "2. 'end' (float, end time in seconds from the beginning) "
            "3. 'text' (string, the spoken text during this segment). "
            "Return ONLY the JSON array. Do not include markdown code block syntax (like ```json). "
            "Provide accurate transcription, maintaining names, terms, and the language of the audio (e.g. Vietnamese/English)."
        )

        model = genai.GenerativeModel("gemini-3.5-flash")
        
        try:
            response = model.generate_content([uploaded_file, prompt])
            text_response = response.text.strip()
            
            # Clean response if markdown blocks are returned
            if text_response.startswith("```"):
                # strip out markdown blocks
                text_response = re.sub(r"^```(?:json)?\n", "", text_response)
                text_response = re.sub(r"\n```$", "", text_response)
                text_response = text_response.strip()
                
            segments = json.loads(text_response)
            
            # Validate structure
            if not isinstance(segments, list):
                raise ValueError("Response is not a list")
                
            # Sort by start time just in case
            segments.sort(key=lambda x: x.get("start", 0))
            
            # Delete remote file to clean up space
            try:
                genai.delete_file(uploaded_file.name)
                print("Cleaned up Gemini remote file.")
            except Exception as delete_err:
                print(f"Warning: Could not delete remote file: {delete_err}")
                
            if progress_callback:
                progress_callback("finished", 100)
                
            return segments
            
        except Exception as e:
            # Try to clean up file even if transcription fails
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
            raise Exception(f"Gemini transcription failed: {str(e)}")

    def _transcribe_openai_whisper(self, audio_path):
        """Transcribes small audio file using OpenAI Whisper API."""
        with open(audio_path, "rb") as audio_file:
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="verbose_json"
            )
            
        # verbose_json returns segments
        segments = []
        if hasattr(transcript, 'segments'):
            for seg in transcript.segments:
                segments.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text")
                })
        else:
            # fallback if segments are not returned directly
            text = getattr(transcript, 'text', '')
            segments.append({
                "start": 0.0,
                "end": 0.0,  # Unknown
                "text": text
            })
            
        return segments
