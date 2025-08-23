#!/usr/bin/env python3
"""
Real-time Audio Transcriber using Whisper
A single-file script that captures microphone or system audio and transcribes it in real-time.

Requirements: uv (https://github.com/astral-sh/uv)
Usage: uv run record.py

To capture SYSTEM AUDIO (instead of microphone):
1. Install BlackHole (virtual audio device):
   brew install blackhole-2ch
   
2. Configure audio routing:
   - Open Audio MIDI Setup (in /Applications/Utilities/)
   - Click '+' at bottom left, create 'Multi-Output Device'
   - Check both 'Built-in Output' and 'BlackHole 2ch'
   - Set this Multi-Output Device as your system output
   
3. Run the script and select BlackHole as input:
   uv run record.py --list-devices  # Find BlackHole device number
   uv run record.py --device <number>

# /// script
# requires-python = ">=3.8,<3.13"
# dependencies = [
#     "openai-whisper==20231117",
#     "pyaudio>=0.2.11",
#     "numpy>=1.24.0,<2.0",
# ]
# ///
"""

import argparse
import queue
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
import ssl
import urllib.request

import numpy as np
import pyaudio
import whisper


class RealTimeTranscriber:
    def __init__(self, model_size="base", chunk_duration=5, device_index=None,
                 beam_size=5, best_of=5, temperature=0.0, auto_device=False, who="Guest"):
        """
        Initialize the real-time transcriber.
        
        Args:
            model_size (str): Whisper model size (tiny, base, small, medium, large)
            chunk_duration (int): Duration of audio chunks to process (seconds)
            device_index (int): Audio device index (None for default)
            beam_size (int): Beam search width for better accuracy (default: 5)
            best_of (int): Number of candidates to consider (default: 5)
            temperature (float): Sampling temperature, 0 for deterministic (default: 0.0)
            auto_device (bool): Automatically find and use BlackHole 2ch device (default: False)
            who (str): Speaker name to display in transcriptions (default: Guest)
        """
        self.model_size = model_size
        self.chunk_duration = chunk_duration
        self.device_index = device_index
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.auto_device = auto_device
        self.who = who
        
        # Load corrections from file if it exists
        import json
        from pathlib import Path
        corrections_file = Path("corrections.json")
        if corrections_file.exists():
            try:
                with open(corrections_file, 'r') as f:
                    self.custom_words = json.load(f)
            except Exception:
                self.custom_words = {}
        else:
            self.custom_words = {}
        
        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1
        self.chunk_size = 1024
        
        # Threading
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.running = False
        
        # Load Whisper model
        # Create SSL context that doesn't verify certificates (for model download)
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
            
        self.model = whisper.load_model(model_size)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Auto-detect BlackHole 2ch if requested
        if self.auto_device and self.device_index is None:
            self.device_index = self.find_blackhole_device()
        
    def find_blackhole_device(self):
        """Find BlackHole 2ch device automatically."""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                # Look specifically for "BlackHole 2ch"
                if "BlackHole 2ch" in info['name']:
                    return i
        return None
    
    def list_audio_devices(self):
        """List available audio input devices."""
        print("\n=== Available Audio Input Devices ===", flush=True)
        print("(To capture system audio, install BlackHole: brew install blackhole-2ch)", flush=True)
        print(flush=True)
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                device_type = ""
                if "BlackHole" in info['name']:
                    device_type = " [Virtual Audio - Can capture system audio]"
                elif "Loopback" in info['name']:
                    device_type = " [Virtual Audio - Can capture system audio]"
                print(f"  {i}: {info['name']} (channels: {info['maxInputChannels']}){device_type}", flush=True)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio callback status: {status}", flush=True)
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data.copy())
        
        return (in_data, pyaudio.paContinue)
    
    def audio_worker(self):
        """Worker thread that collects audio chunks and queues them for transcription."""
        audio_buffer = np.array([], dtype=np.float32)
        chunk_samples = self.sample_rate * self.chunk_duration
        
        while self.running:
            try:
                # Get audio data from queue (timeout to check if still running)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.append(audio_buffer, audio_chunk)
                
                # Process when we have enough audio
                if len(audio_buffer) >= chunk_samples:
                    # Take the required amount and keep the rest
                    to_process = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples:]
                    
                    # Queue for transcription
                    self.transcription_queue.put(to_process)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio worker: {e}", flush=True)
    
    def transcription_worker(self):
        """Worker thread that transcribes queued audio chunks."""
        while self.running:
            try:
                # Get audio chunk to transcribe
                audio_chunk = self.transcription_queue.get(timeout=0.1)
                
                # Transcribe with Whisper
                start_time = time.time()
                result = self.model.transcribe(
                    audio_chunk,
                    fp16=False,  # Use fp32 for CPU compatibility
                    language="en",  # English only
                    task="transcribe",
                    beam_size=self.beam_size,  # Use beam search for better accuracy
                    best_of=self.best_of,  # Consider multiple candidates
                    temperature=self.temperature,  # Sampling temperature
                    compression_ratio_threshold=2.4,  # Filter out nonsense
                    no_speech_threshold=0.6,  # Better silence detection
                    condition_on_previous_text=False  # Fresh context each chunk
                )
                
                transcription_time = time.time() - start_time
                text = result["text"].strip()
                
                # Apply custom word replacements
                text = self.apply_custom_words(text)
                
                if text:  # Only print non-empty transcriptions
                    timestamp = datetime.now().strftime("%H:%M")
                    print(f"[{timestamp}] {self.who}: {text}", flush=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in transcription worker: {e}", flush=True)
    
    def start_transcription(self):
        """Start real-time transcription."""
        
        # Start audio stream
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}", flush=True)
            return
        
        # Start worker threads
        self.running = True
        
        audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        
        audio_thread.start()
        transcription_thread.start()
        
        # Start audio stream
        stream.start_stream()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping transcription...", flush=True)
            
        # Cleanup
        self.running = False
        stream.stop_stream()
        stream.close()
        
        # Wait for threads to finish
        audio_thread.join(timeout=1.0)
        transcription_thread.join(timeout=1.0)
        
        print("Transcription stopped.", flush=True)
    
    def apply_custom_words(self, text):
        """Apply custom word replacements for better name recognition."""
        if not self.custom_words:
            return text
        
        import re
        for correct_word, variations in self.custom_words.items():
            for variant in variations:
                # Case-insensitive replacement while preserving the case style
                pattern = re.compile(re.escape(variant), re.IGNORECASE)
                text = pattern.sub(correct_word, text)
        
        return text
    
    def __del__(self):
        """Cleanup PyAudio."""
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using OpenAI Whisper"
    )
    parser.add_argument(
        "--model", 
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size. Larger = better quality but slower (default: small)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Audio chunk duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Audio input device index (use --list-devices to see options)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0 for deterministic (default: 0.0)"
    )
    parser.add_argument(
        "--auto-device",
        action="store_true",
        help="Automatically detect and use BlackHole 2ch device for system audio capture"
    )
    parser.add_argument(
        "--who",
        type=str,
        default="Guest",
        help="Speaker name to display in transcriptions (default: Guest, use 'Me' for yourself)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search width for better accuracy (default: 5)"
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates to consider (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create transcriber
    transcriber = RealTimeTranscriber(
        model_size=args.model,
        chunk_duration=args.duration,
        device_index=args.device,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        auto_device=args.auto_device,
        who=args.who
    )
    
    if args.list_devices:
        transcriber.list_audio_devices()
        return
    
    # Start transcription
    try:
        transcriber.start_transcription()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
