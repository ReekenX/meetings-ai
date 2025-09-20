# Meeting Recording Instructions

## Starting a New Meeting Recording

When the user wants to start a new meeting recording:

1. Run the following command in the background:

   ```bash
   tmux send-keys -t meeting:transcript "tail -f [NAME]" C-m;
   uv run record.py --model tiny --duration 5 --beam-size 1 --best-of 1 | tee [NAME]
   ```

   Where `[NAME]` is the file path provided by the user (e.g., `meetings/group/2025-08-22-meeting-title.md`)

2. Keep this command running until the user tells you to stop

## Stopping the Recording

When the user says to stop:

- Kill the background process running the recording command

## Post-Recording Tasks

After stopping the recording:

1. Review the generated file (e.g., `meetings/group/2025-08-22-meeting-title.md`)
2. Update all `[TODO]` placeholders in the file:
   - Summary
   - Goal
   - Title
   - Any other TODO markers
3. Ask the user to provide meeting participants
4. Update the file with the participant information

## Answering Questions About Meetings

When the user asks questions about a meeting:

1. **Always re-read the meeting file** (the `[NAME]` file from the recording) before answering any question
2. Answer directly from the content of the meeting transcript
3. **Do not use any external tools or internet search** - only use the information from the meeting file
4. Reference specific parts of the transcript when answering

