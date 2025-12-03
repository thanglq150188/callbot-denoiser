"""
Audio VAD processor using Silero VAD (PyTorch)

This module provides Voice Activity Detection for mono audio files.
Assumes input audio is already single channel (mono).

Original source: src/core/audio_processor.py
"""

import torch
import librosa
from pydub import AudioSegment
import os


# Global model cache (lazy loaded)
_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None
_VAD_LOADED = False


def _load_silero_vad():
    """Load a fresh Silero VAD model and utils instance using PyTorch"""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        onnx=True,
        force_reload=False  # Use cache but create new instance
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def load_vad_models():
    """Load VAD models lazily (call this before using VAD functions)"""
    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS, _VAD_LOADED

    if _VAD_LOADED:
        return

    print("Loading VAD model (PyTorch)...")
    _VAD_MODEL, _GET_SPEECH_TIMESTAMPS = _load_silero_vad()
    print("PyTorch VAD model loaded successfully!")
    _VAD_LOADED = True


def detect_speech_vads_fn(
    audio_path: str,
    output_folder: str,
    sample_rate=8000,
    threshold=0.2,
    min_speech_duration_ms=50,
    min_silence_duration_ms=100,
    speech_pad_ms=30
) -> dict:
    """
    Detect speech chunks in a mono audio file using Silero VAD (PyTorch) and save them

    Args:
        audio_path: Path to mono audio file (mp3, wav, etc.)
        output_folder: Folder to save detected speech chunks
        sample_rate: Sample rate for VAD (default 8000)
        threshold: Speech threshold (0.0-1.0). Higher = more strict. Default 0.5
        min_speech_duration_ms: Minimum duration of speech chunk in ms. Default 50ms
        min_silence_duration_ms: Minimum duration of silence between chunks in ms. Default 100ms
        speech_pad_ms: Padding to add before/after each speech chunk in ms. Default 30ms

    Returns:
        dict:
        - vads(list[dict]): List of detected speech chunks
    """
    # Ensure VAD models are loaded
    load_vad_models()

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load audio file
    audio = AudioSegment.from_file(audio_path)

    # Load with librosa for VAD
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Convert to torch tensor
    waveform_tensor = torch.FloatTensor(waveform)

    # Perform VAD with custom parameters
    vad_results = _GET_SPEECH_TIMESTAMPS(
        waveform_tensor,
        _VAD_MODEL,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )

    # Get base filename for output chunks
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    # Process each chunk: convert to ms, extract, and save
    chunks = []
    for i, chunk in enumerate(vad_results):
        start_ms = chunk['start'] / sample_rate * 1000
        end_ms = chunk['end'] / sample_rate * 1000
        duration_ms = (chunk['end'] - chunk['start']) / sample_rate * 1000

        # Extract audio chunk
        audio_chunk = audio[start_ms:end_ms]

        # Generate output filename
        chunk_filename = f"{base_filename}_chunk_{i+1:03d}_{int(start_ms)}ms_{int(end_ms)}ms.wav"
        chunk_path = os.path.join(output_folder, chunk_filename)

        # Save chunk
        audio_chunk.export(chunk_path, format="wav")

        chunks.append({
            'start': start_ms / 1000,
            'end': end_ms / 1000,
            'duration': duration_ms / 1000,
            'file_path': chunk_path
        })

    return {"vads": chunks}



if __name__ == "__main__":
    # Example usage
    audio_path = "step2_final_gated.wav"
    output_dir = "output"

    # Detect VADs in mono audio file
    vads_result = detect_speech_vads_fn(
        audio_path=audio_path,
        output_folder=os.path.join(output_dir, "chunks")
    )
    vads = vads_result['vads']

    # Print detected VADs
    for vad in vads:
        print(vad)
