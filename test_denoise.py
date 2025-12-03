import os
import numpy as np
import soundfile as sf
import librosa
import torch
from pydub import AudioSegment


INPUT_FILE = r'D:\callbot-denoiser\denoise_data\samples\11a72914-ebfa-4115-bd23-0b968a88c837.mp3'
OUTPUT_DIR = r'D:\callbot-denoiser\output'


# Global VAD model cache
_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None
_VAD_LOADED = False


def _load_silero_vad():
    """Load Silero VAD model"""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        onnx=True,
        force_reload=False
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def load_vad_models():
    """Load VAD models lazily"""
    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS, _VAD_LOADED
    if _VAD_LOADED:
        return
    print("  Loading VAD model...")
    _VAD_MODEL, _GET_SPEECH_TIMESTAMPS = _load_silero_vad()
    _VAD_LOADED = True


def extract_left_channel(audio_path, output_path):
    """
    Extract the left channel from a stereo audio file.
    """
    print(f"\n[Step 1] Extracting Left Channel")
    
    # Load audio with librosa (handles MP3)
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    if audio.ndim == 1:
        print("  Audio is mono, using as-is")
        left_channel = audio
    else:
        print(f"  Audio has {audio.shape[0]} channels, extracting left (channel 0)")
        left_channel = audio[0]
    
    # Save as WAV
    sf.write(output_path, left_channel, sr)
    print(f"  Duration: {len(left_channel) / sr:.2f}s")
    print(f"  Saved to: {output_path}")
    return output_path, sr


def noise_gate(audio_path, output_path, threshold_db=-35):
    """
    Apply noise gate to silence audio below threshold.
    """
    print(f"\n[Step 2] Applying Noise Gate")
    print(f"  Threshold: {threshold_db} dB")
    
    audio, sr = sf.read(audio_path)
    
    window_size = int(sr * 0.02)  # 20ms window
    silenced_windows = 0
    total_windows = 0
    
    for i in range(0, len(audio), window_size):
        chunk = audio[i:i+window_size]
        if len(chunk) == 0:
            continue
        total_windows += 1
        rms = np.sqrt(np.mean(chunk**2))
        db = 20 * np.log10(rms + 1e-10)
        
        if db < threshold_db:
            audio[i:i+window_size] = 0
            silenced_windows += 1
    
    sf.write(output_path, audio, sr)
    print(f"  Silenced {silenced_windows}/{total_windows} windows ({100*silenced_windows/total_windows:.1f}%)")
    print(f"  Saved to: {output_path}")
    return output_path


def detect_vads(
    audio_path: str,
    output_folder: str,
    sample_rate=8000,
    threshold=0.2,
    min_speech_duration_ms=50,
    min_silence_duration_ms=100,
    speech_pad_ms=30
) -> list:
    """
    Detect speech chunks using Silero VAD and save them as separate files.
    
    Returns:
        List of dicts with start, end, duration, file_path for each chunk.
    """
    print(f"\n[Step 3] Detecting Voice Activity")
    
    load_vad_models()
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio for chunking (original sample rate)
    audio = AudioSegment.from_file(audio_path)
    
    # Load with librosa for VAD (resampled)
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform_tensor = torch.FloatTensor(waveform)
    
    # Perform VAD
    vad_results = _GET_SPEECH_TIMESTAMPS(
        waveform_tensor,
        _VAD_MODEL,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    chunks = []
    for i, chunk in enumerate(vad_results):
        start_ms = chunk['start'] / sample_rate * 1000
        end_ms = chunk['end'] / sample_rate * 1000
        duration_ms = end_ms - start_ms
        
        # Extract and save chunk
        audio_chunk = audio[start_ms:end_ms]
        chunk_filename = f"{base_filename}_chunk_{i+1:03d}_{int(start_ms)}ms_{int(end_ms)}ms.wav"
        chunk_path = os.path.join(output_folder, chunk_filename)
        audio_chunk.export(chunk_path, format="wav")
        
        chunks.append({
            'start': start_ms / 1000,
            'end': end_ms / 1000,
            'duration': duration_ms / 1000,
            'file_path': chunk_path
        })
    
    print(f"  Found {len(chunks)} speech segments")
    print(f"  Saved to: {output_folder}")
    return chunks


def process_audio(
    input_file: str,
    output_dir: str,
    threshold_db: float = -35,
    vad_threshold: float = 0.2
) -> dict:
    """
    Full audio processing pipeline: Extract Left Channel -> Noise Gate -> VAD
    
    Args:
        input_file: Path to input audio file (MP3, WAV, etc.)
        output_dir: Directory to save output files
        threshold_db: Noise gate threshold in dB
        vad_threshold: VAD speech detection threshold (0.0-1.0)
    
    Returns:
        Dict with paths to intermediate files and VAD chunks
    """
    print("=" * 60)
    print("Audio Processing Pipeline")
    print("  Left Channel -> Noise Gate -> VAD")
    print("=" * 60)
    print(f"Input: {input_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract left channel
    left_channel_path = os.path.join(output_dir, "step1_left_channel.wav")
    extract_left_channel(input_file, left_channel_path)
    
    # Step 2: Apply noise gate
    gated_path = os.path.join(output_dir, "step2_gated.wav")
    noise_gate(left_channel_path, gated_path, threshold_db=threshold_db)
    
    # Step 3: Detect VADs
    chunks_folder = os.path.join(output_dir, "chunks")
    vads = detect_vads(gated_path, chunks_folder, threshold=vad_threshold)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {left_channel_path} (left channel)")
    print(f"  2. {gated_path} (noise gated)")
    print(f"  3. {chunks_folder}/ ({len(vads)} VAD chunks)")
    
    return {
        'left_channel': left_channel_path,
        # 'gated': gated_path,
        'chunks_folder': chunks_folder,
        'vads': vads
    }


def main():
    result = process_audio(INPUT_FILE, OUTPUT_DIR)
    
    # Print VAD results
    print(f"\nVAD Segments:")
    for i, vad in enumerate(result['vads']):
        print(f"  {i+1}. {vad['start']:.2f}s - {vad['end']:.2f}s ({vad['duration']:.2f}s)")


if __name__ == "__main__":
    main()
