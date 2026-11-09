#!/bin/sh
# usage: run_chrome.sh <page> <outfile>   (headless Chrome, local http server on 8765)
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
S=$(date +%s)
"$C" --headless=new --disable-gpu --user-data-dir="${TMPDIR:-/tmp}/chprof_$$" --dump-dom "http://127.0.0.1:8765/web/$1" > "$2" 2>/dev/null
echo "chrome_wall_seconds=$(( $(date +%s) - S ))" >> "$2.time"
