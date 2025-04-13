import os
import argparse
import requests
import concurrent.futures
from pydub import AudioSegment

def split_audio(audio_path, chunk_length_minutes=10):
    """
    Splits the audio file into chunks of specified minutes.
    Each chunk is exported as an MP3 file to a separate folder.
    Returns a list of chunk file paths.
    """
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

def transcribe_chunk(chunk_file, chunk_index, api_key, model="whisper-1"):
    """
    Sends an API request to OpenAI’s transcription endpoint for a given chunk.
    Returns a tuple of (chunk_index, transcript_text).
    """
    print(f"Starting transcription for chunk {chunk_index+1}: {chunk_file}")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model
        # Optionally, add additional parameters such as language if needed.
    }
    try:
        with open(chunk_file, "rb") as f:
            files = {
                "file": (os.path.basename(chunk_file), f, "audio/mpeg")
            }
            response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        transcript = response.json().get("text", "")
        print(f"Chunk {chunk_index+1} transcription completed.")
        return (chunk_index, transcript)
    except Exception as e:
        print(f"Error transcribing chunk {chunk_index+1}: {e}")
        return (chunk_index, "")

def main():
    parser = argparse.ArgumentParser(
        description="Split a large audio file, transcribe each chunk in parallel using the OpenAI API, and stitch the transcripts together."
    )
    parser.add_argument("audio_file", type=str, help="Path to the large audio file (MP3, WAV, etc.).")
    parser.add_argument("--chunk_length", type=int, default=10, help="Chunk length in minutes (default=10).")
    parser.add_argument("--model", type=str, default="whisper-1",
                        help="API model to use for transcription (default: whisper-1).")
    args = parser.parse_args()

    # Validate API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit(1)

    # Split the audio file into chunks
    print("Splitting audio file into chunks...")
    chunks = split_audio(args.audio_file, chunk_length_minutes=args.chunk_length)
    total_chunks = len(chunks)
    print(f"Audio file split into {total_chunks} chunks.")

    # Transcribe chunks in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx, chunk in enumerate(chunks):
            futures.append(executor.submit(transcribe_chunk, chunk, idx, api_key, args.model))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    # Sort transcripts by the original chunk order
    results.sort(key=lambda x: x[0])
    transcripts = [text for idx, text in results if text]

    # Stitch all transcripts together into one complete transcript
    combined_transcript = "\n\n".join(transcripts)
    output_file = f"{os.path.splitext(args.audio_file)[0]}_transcription.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_transcript)

    print(f"\nTranscription complete. Output saved to '{output_file}'.")

if __name__ == "__main__":
    main()

