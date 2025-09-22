# Audio Transcription Suite for macOS

Real-time audio transcription tools for capturing and transcribing system audio (Zoom meetings, etc.).

## Features

- 🎙️ **Real-time transcription** using OpenAI Whisper
- 🔊 **System audio capture** (Zoom, Teams, Google Hangout, etc.)
- 🚀 **Auto-device detection** for virtual audio devices
- 💻 **No cloud dependencies** - runs entirely locally

## Quick Start

### System Audio Setup

To capture system audio (required for Zoom/Teams/etc.), you need a virtual audio device:

#### BlackHole Setup (Free)

```bash
# Install BlackHole
brew install blackhole-2ch

# Configure audio routing
# 1. Open Audio MIDI Setup (/Applications/Utilities/)
# 2. Click '+' → 'Create Multi-Output Device'
# 3. Check both:
#    ✓ Built-in Output (to hear audio)
#    ✓ BlackHole 2ch (to capture audio)
# 4. Set Multi-Output Device as system output
```

## Usage

### 1. Basic Microphone Transcription

```bash
# Using uv (recommended)
uv run record.py

# Or with Python directly
python3 record.py
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - Use freely for personal and commercial purposes.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for transcription models
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) for virtual audio