# Meeting Recording Instructions

## Available Commands

- `/meeting:start [TITLE]` - Start a new meeting recording with the given title
- `/meeting:stop` - Stop the current recording

For detailed command documentation, see:
- `.claude/commands/meeting:start.md`
- `.claude/commands/meeting:stop.md`

## Answering Questions About Meetings

When the user asks questions about a meeting:

1. Before answering any question, **always re-read the meeting transcript** (the `cat $(cat .current-meeting)` bash command will output meeting transcript)
2. Answer directly from the contents of the meeting transcript
3. **Do not use any external tools or internet search** - only use the information from the meeting file
4. Reference specific parts of the transcript when answering

