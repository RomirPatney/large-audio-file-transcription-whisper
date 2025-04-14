# large-audio-file-transcription-whisper
Quick script to transcribe large audio file to text

# Setup
brew install python@3.11 ffmpeg

cd your_project_folder
python3.11 -m venv venv

source venv/bin/activate

pip install -U pip wheel setuptools

pip install openai-whisper torch torchvision torchaudio ffmpeg-python

# Offline transcription
python3.11 ./offline-transcribe.py "audio.mp3"

python3.11 ./offline-transcribe.py "audio.mp3" --chunk_length 15 --model small


# Online transcription
export OPENAI_API_KEY="sk-yourkeyhere"

echo "$OPENAI_API_KEY"

python3.11 ./online-transcribe.py "audio.mp3"

python3.11 ./online-transcribe.py "audio.mp3" --chunk_length 15 --model whisper-1
