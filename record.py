#!/usr/bin/env python3
"""
Dual-Source Real-time Audio Transcriber using Whisper
Automatically transcribes from both BlackHole 2ch (Guest) and AirPods microphone (Me).

Requirements: uv (https://github.com/astral-sh/uv)
Usage: uv run record_dual.py

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
from datetime import datetime
from pathlib import Path
import ssl
import json

import numpy as np
import pyaudio
import whisper


class DualSourceTranscriber:
    def __init__(self, model_size="small", chunk_duration=3, 
                 beam_size=1, best_of=1, temperature=0.0, no_header=False):
        """
        Initialize the dual-source transcriber.
        
        Args:
            model_size (str): Whisper model size (tiny, base, small, medium, large)
            chunk_duration (int): Duration of audio chunks to process (seconds)
            beam_size (int): Beam search width for better accuracy
            best_of (int): Number of candidates to consider
            temperature (float): Sampling temperature, 0 for deterministic
            no_header (bool): Skip printing the meeting header
        """
        self.model_size = model_size
        self.chunk_duration = chunk_duration
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.no_header = no_header
        
        # Load corrections from file if it exists
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
        
        # Source configurations
        self.sources = []
        
        # Threading
        self.running = False
        
        # Load Whisper models - one for each source to avoid conflicts
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
            
        # We'll load models when we know how many sources we have
        self.model_size_cached = model_size
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Auto-detect devices
        self.setup_audio_sources()
    
    def find_device_by_name(self, target_names):
        """Find audio device by name patterns."""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                for target in target_names:
                    if target.lower() in info['name'].lower():
                        return i, info['name'], info['maxInputChannels']
        return None, None, None
    
    def setup_audio_sources(self):
        """Setup audio sources for BlackHole and AirPods."""
        # Find BlackHole 2ch for Guest
        blackhole_idx, blackhole_name, blackhole_channels = self.find_device_by_name(["blackhole 2ch"])
        if blackhole_idx is not None:
            print(f"Loading Whisper model for Guest...", flush=True)
            model = whisper.load_model(self.model_size_cached)
            self.sources.append({
                'device_index': blackhole_idx,
                'device_name': blackhole_name,
                'channels': min(blackhole_channels, 2),  # Use up to 2 channels
                'who': 'Guest',
                'model': model,  # Dedicated model instance
                'model_lock': threading.Lock(),  # Lock for thread-safe model access
                'audio_queue': queue.Queue(),
                'transcription_queue': queue.Queue(),
                'stream': None,
                'audio_thread': None,
                'transcription_thread': None
            })
            print(f"✓ Found BlackHole 2ch (device {blackhole_idx}, {blackhole_channels} channels) for Guest transcription", flush=True)
        else:
            print("⚠ BlackHole 2ch not found - Guest audio will not be captured", flush=True)
        
        # Find AirPods for Me
        airpods_idx, airpods_name, airpods_channels = self.find_device_by_name(["airpods pro"])
        if airpods_idx is not None:
            print(f"Loading Whisper model for Me...", flush=True)
            model = whisper.load_model(self.model_size_cached)
            self.sources.append({
                'device_index': airpods_idx,
                'device_name': airpods_name,
                'channels': airpods_channels,  # Use actual channel count
                'who': 'Me',
                'model': model,  # Dedicated model instance
                'model_lock': threading.Lock(),  # Lock for thread-safe model access
                'audio_queue': queue.Queue(),
                'transcription_queue': queue.Queue(),
                'stream': None,
                'audio_thread': None,
                'transcription_thread': None
            })
            print(f"✓ Found AirPods Pro II (device {airpods_idx}, {airpods_channels} channel) for Me transcription", flush=True)
        else:
            print("⚠ AirPods Pro II not found - Your audio will not be captured", flush=True)
        
        if not self.sources:
            print("\n❌ No audio devices found. Please ensure:", flush=True)
            print("   - BlackHole 2ch is installed and configured for system audio", flush=True)
            print("   - AirPods Pro II are connected", flush=True)
            sys.exit(1)
    
    def audio_callback(self, source_idx):
        """Create audio callback for a specific source."""
        def callback(in_data, frame_count, time_info, status):
            if status:
                print(f"Audio callback status for {self.sources[source_idx]['who']}: {status}", flush=True)
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Convert to mono if multi-channel
            channels = self.sources[source_idx]['channels']
            if channels > 1:
                # Reshape to (frames, channels) and average across channels
                audio_data = audio_data.reshape(-1, channels)
                audio_data = np.mean(audio_data, axis=1)
            
            self.sources[source_idx]['audio_queue'].put(audio_data.copy())
            
            return (in_data, pyaudio.paContinue)
        return callback
    
    def audio_worker(self, source_idx):
        """Worker thread that collects audio chunks for a specific source."""
        source = self.sources[source_idx]
        audio_buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.sample_rate * self.chunk_duration)  # Ensure integer
        
        while self.running:
            try:
                # Get audio data from queue
                audio_chunk = source['audio_queue'].get(timeout=0.1)
                audio_buffer = np.append(audio_buffer, audio_chunk)
                
                # Process when we have enough audio
                if len(audio_buffer) >= chunk_samples:
                    # Take exactly the required amount and keep the rest
                    to_process = audio_buffer[:chunk_samples].astype(np.float32)
                    audio_buffer = audio_buffer[chunk_samples:]
                    
                    # Ensure the audio chunk is exactly the right size
                    if len(to_process) != chunk_samples:
                        print(f"Warning: Audio chunk size mismatch for {source['who']}: {len(to_process)} != {chunk_samples}", flush=True)
                        continue
                    
                    # Queue for transcription
                    source['transcription_queue'].put(to_process)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio worker for {source['who']}: {e}", flush=True)
    
    def transcription_worker(self, source_idx):
        """Worker thread that transcribes audio for a specific source."""
        source = self.sources[source_idx]
        expected_samples = int(self.sample_rate * self.chunk_duration)
        
        while self.running:
            try:
                # Get audio chunk to transcribe
                audio_chunk = source['transcription_queue'].get(timeout=0.1)
                
                # Validate audio chunk size
                if len(audio_chunk) != expected_samples:
                    print(f"Skipping invalid audio chunk for {source['who']}: {len(audio_chunk)} samples (expected {expected_samples})", flush=True)
                    continue
                
                # Ensure float32 and correct shape
                audio_chunk = audio_chunk.astype(np.float32)
                
                # Check if audio contains actual sound (not just silence)
                # Calculate RMS (root mean square) to detect silence
                rms = np.sqrt(np.mean(audio_chunk**2))
                if rms < 0.002:  # Increased threshold for better silence detection
                    continue  # Skip silent chunks
                
                # Additional check: if audio is too uniform (likely silence or noise)
                audio_std = np.std(audio_chunk)
                if audio_std < 0.001:
                    continue
                
                # Transcribe with source-specific Whisper model (with lock for thread safety)
                with source['model_lock']:
                    result = source['model'].transcribe(
                        audio_chunk,
                        fp16=False,
                        language="en",
                        task="transcribe",
                        beam_size=self.beam_size,
                        best_of=self.best_of,
                        temperature=self.temperature,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False
                    )
                
                text = result["text"].strip()
                
                # Check for common Whisper hallucinations in silence
                hallucination_patterns = [
                    "thank you", "thanks", "you", "bye", "goodbye", 
                    "thank you for watching", "please subscribe",
                    "see you next time", "bye-bye", "thank you.",
                    "thank you so much", "thank you very much"
                ]
                
                # More aggressive filtering for common hallucinations
                text_lower = text.lower().strip('.')
                if text_lower in hallucination_patterns or text_lower.startswith("thank you"):
                    # Always skip "thank you" variants unless we have very high confidence it's real speech
                    if 'no_speech_prob' in result and result['no_speech_prob'] > 0.2:
                        continue
                    # Even with low no_speech_prob, skip if RMS is very low (near silence)
                    if rms < 0.005:
                        continue
                
                # Skip very short repetitive text that might be hallucinations
                if len(text.split()) <= 3 and 'no_speech_prob' in result and result['no_speech_prob'] > 0.25:
                    continue
                
                # Additional filter: skip if segments show high no_speech probability
                if 'segments' in result:
                    segments_no_speech = [seg.get('no_speech_prob', 0) for seg in result['segments'] if 'no_speech_prob' in seg]
                    if segments_no_speech and np.mean(segments_no_speech) > 0.4:
                        continue
                
                # Apply custom word replacements
                text = self.apply_custom_words(text)
                
                if text:  # Only print non-empty transcriptions
                    timestamp = datetime.now().strftime("%H:%M")
                    print(f"[{timestamp}] {source['who']}: {text}", flush=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in transcription worker for {source['who']}: {e}", flush=True)
    
    def apply_custom_words(self, text):
        """Apply custom word replacements for better name recognition."""
        if not self.custom_words:
            return text
        
        import re
        for correct_word, variations in self.custom_words.items():
            for variant in variations:
                pattern = re.compile(re.escape(variant), re.IGNORECASE)
                text = pattern.sub(correct_word, text)
        
        return text
    
    def print_meeting_header(self):
        """Print the meeting header template."""
        header = """# [TITLE TODO]

## Goal

[ONE SENTENCES GOAL TODO]

## Participants

[PARTICIPANTS TODO]

## Summary

[SUMMARY GROUP TODO]
- [SUMMARY ITEM TODO]
- [SUMMARY ITEM TODO]

[ANOTHER GROUP TODO]
- [ANOTHER ITEM TODO]
- [ANOTHER ITEM TODO]

## Transcript

```"""
        print(header, flush=True)
    
    def start_transcription(self):
        """Start dual-source transcription."""
        
        # Print meeting header unless disabled
        if not self.no_header:
            self.print_meeting_header()
        
        print(f"\nStarting dual-source transcription...", flush=True)
        print(f"Listening to {len(self.sources)} audio source(s):", flush=True)
        for source in self.sources:
            print(f"  - {source['who']}: {source['device_name']}", flush=True)
        print("\nPress Ctrl+C to stop recording.\n", flush=True)
        
        # Start audio streams for each source
        for idx, source in enumerate(self.sources):
            try:
                stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=source['channels'],  # Use device-specific channel count
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=source['device_index'],
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self.audio_callback(idx)
                )
                source['stream'] = stream
            except Exception as e:
                print(f"Error opening audio stream for {source['who']}: {e}", flush=True)
                continue
        
        # Start worker threads for each source
        self.running = True
        
        for idx, source in enumerate(self.sources):
            if source['stream'] is not None:
                audio_thread = threading.Thread(
                    target=self.audio_worker, 
                    args=(idx,), 
                    daemon=True
                )
                transcription_thread = threading.Thread(
                    target=self.transcription_worker, 
                    args=(idx,), 
                    daemon=True
                )
                
                source['audio_thread'] = audio_thread
                source['transcription_thread'] = transcription_thread
                
                audio_thread.start()
                transcription_thread.start()
                source['stream'].start_stream()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping transcription...", flush=True)
        
        # Cleanup
        self.running = False
        
        for source in self.sources:
            if source['stream'] is not None:
                source['stream'].stop_stream()
                source['stream'].close()
        
        # Wait for threads to finish
        for source in self.sources:
            if source['audio_thread'] is not None:
                source['audio_thread'].join(timeout=1.0)
            if source['transcription_thread'] is not None:
                source['transcription_thread'].join(timeout=1.0)
        
        print("Transcription stopped.", flush=True)
    
    def __del__(self):
        """Cleanup PyAudio."""
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Dual-source real-time audio transcription using OpenAI Whisper"
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
        default=3,
        help="Audio chunk duration in seconds (default: 3)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0 for deterministic (default: 0.0)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam search width for better accuracy (default: 1)"
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Number of candidates to consider (default: 1)"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Skip printing the meeting header template"
    )
    
    args = parser.parse_args()
    
    # Create dual-source transcriber
    transcriber = DualSourceTranscriber(
        model_size=args.model,
        chunk_duration=args.duration,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        no_header=args.no_header
    )
    
    # Start transcription
    try:
        transcriber.start_transcription()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()