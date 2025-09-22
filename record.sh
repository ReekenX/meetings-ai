#!/bin/bash

TITLE=$*
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//');
DATE=$(date +%Y-%m-%d);

echo "# $TITLE" > "meetings/$DATE-$SLUG.md";
echo "" >> "meetings/$DATE-$SLUG.md";
echo "meetings/$DATE-$SLUG.md" > .current-meeting;
tmux send-keys -t meetings-ai:transcript "tail -f 'meetings/$DATE-$SLUG.md'" C-m;
uv run record.py --model tiny --duration 2 --fp16 | tee "meetings/$DATE-$SLUG.md"