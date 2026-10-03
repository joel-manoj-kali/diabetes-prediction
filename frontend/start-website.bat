@echo off
cd /d "%~dp0"
echo Starting the diabetes risk website at http://localhost:8080
echo Keep this window open while using the website.
start "" http://localhost:8080
python -m http.server 8080