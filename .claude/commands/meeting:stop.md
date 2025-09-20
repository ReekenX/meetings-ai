# /meeting:stop

Stop the current meeting recording and process the transcript

## Usage

```
/meeting:stop
```

## Description

This command stops an active meeting recording session and performs post-processing tasks on the transcript.

## What it does

1. Stops the background recording process
2. Reviews the generated transcript file
3. Updates placeholder information in the transcript
4. Prompts for participant information
5. Finalizes the meeting transcript

## Implementation

When this command is invoked:

1. **Stop the recording:**
   - Find and kill the background process running `record.py`
   - Confirm the recording has stopped

2. **Post-processing tasks:**
   - Read the generated transcript file
   - Identify and update all `[TODO]` placeholders:
     - Summary: Generate a concise meeting summary
     - Goal: Extract the main goal or purpose of the meeting
     - Title: Update with a descriptive title
     - Any other TODO markers found in the transcript

3. **Participant information:**
   - Ask the user to provide the list of meeting participants
   - Update the transcript with participant names

4. **Finalize:**
   - Save all updates to the transcript file
   - Inform the user that the recording has been stopped and processed

## Notes

- This command will only work if there's an active recording started with `/meeting:start`
- The transcript file is automatically saved and updated
- All TODO placeholders are replaced with actual content based on the meeting discussion