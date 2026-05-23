#!/bin/bash
# Run this ONCE on the live Linux server to install and start the face service
set -e

cd "$(dirname "$0")"

echo "=== Installing Python dependencies (no dlib, no compilation) ==="
pip3 install -r requirements.txt

echo "=== Starting face recognition service in background ==="
nohup python3 -m uvicorn main:app --host 127.0.0.1 --port 8001 > face_service.log 2>&1 &
echo $! > face_service.pid

echo "=== Service started. PID: $(cat face_service.pid) ==="
echo "=== Models will be auto-downloaded on first request (~30MB) ==="
echo "=== Check logs: tail -f face_service.log ==="
echo "=== Stop service: kill \$(cat face_service.pid) ==="
