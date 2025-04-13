import os
import argparse
from pydub import AudioSegment
import whisper
import math

def split_audio(audio_path, chunk_length_minutes=10):
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_length_minutes * 60 * 1000

    chunks_dir = f"{os.path.splitext(audio_path)[0]}_chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_name = os.path.join(chunks_dir, f"chunk_{i//chunk_length_ms}.mp3")
        chunk.export(chunk_name, format="mp3")
        chunks.append(chunk_name)

    return chunks

def transcribe_chunks(chunks, model_size="medium"):
    model = whisper.load_model(model_size)
    transcripts = []

    for idx, chunk in enumerate(chunks):
        print(f"Transcribing chunk {idx + 1}/{len(chunks)}: {chunk}")
        result = model.transcribe(chunk, fp16=False)
        transcripts.append(result["text"])

    return transcripts

def main():
    parser = argparse.ArgumentParser(description="Split and transcribe large audio files into manageable chunks.")
    parser.add_argument("audio_file", type=str, help="Path to the large audio file (MP3, WAV, etc.).")
    parser.add_argument("--chunk_length", type=int, default=10, help="Chunk length in minutes (default=10).")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium).")

    args = parser.parse_args()

    chunks = split_audio(args.audio_file, chunk_length_minutes=args.chunk_length)
    transcripts = transcribe_chunks(chunks, model_size=args.model)

    # Combine all transcriptions into one file
    combined_transcript = "\n\n".join(transcripts)
    output_transcript_path = f"{os.path.splitext(args.audio_file)[0]}_transcription.txt"
    
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        f.write(combined_transcript)

    print(f"\nTranscription completed successfully. Final transcript saved as '{output_transcript_path}'.")

if __name__ == "__main__":
    main()

