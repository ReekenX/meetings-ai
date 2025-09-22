# /meeting:start

Start a new meeting recording with automatic transcription

## Usage

```
/meeting:start [TITLE]
```

## Description

This command starts a new meeting recording session with real-time transcription using Whisper AI model.

## Parameters

- `TITLE` (required): The title/name for the meeting recording. This will be used as the filename for the transcript.
  - Example: `Team Standup` will be slugified for the filename as `team-standup`
  - The file will be saved as `meetings/[YYYY-MM-DD]-[SLUG].md`

## What it does

1. Creates a new meeting transcript file at `meetings/[YYYY-MM-DD]-[SLUG].md`
2. Starts recording audio from your microphone
3. Transcribes the audio in real-time using Whisper (tiny model)
4. Continuously appends the transcription to the file
5. Displays the transcript in a tmux pane for real-time viewing

## Implementation

When this command is invoked:

1. Create the meetings directory if it doesn't exist
2. Construct the full file path: `meetings/[YYYY-MM-DD]-[SLUG].md`
3. Run the following command in the background:
   ```bash
   ./record.sh [TITLE]
   ```
4. Store the background process ID for later stopping
5. Inform the user that recording has started

## Notes

- The recording continues until explicitly stopped with `/meeting:stop`
- Audio is processed in 2-second chunks for near real-time transcription
- Uses the tiny Whisper model for fast processing
- FP16 precision is used for improved performance