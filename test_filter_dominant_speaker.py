import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


def compute_frame_energy(
    audio: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """Compute short-term energy per frame using RMS (more stable than sum of squares)."""
    # Use librosa's RMS for efficiency
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def compute_global_energy_percentile(
    energy: np.ndarray,
    percentile: float = 75
) -> float:
    """
    Compute energy threshold based on global percentile.
    This identifies the "main speaker" level across the entire audio.
    """
    # Only consider non-silent frames for percentile calculation
    non_silent = energy[energy > np.max(energy) * 0.001]  # Ignore very quiet frames
    if len(non_silent) == 0:
        return 0.0
    return np.percentile(non_silent, percentile)


def filter_dominant_speaker(
    audio: np.ndarray,
    sr: int = 16000,
    frame_length: int = 512,
    hop_length: int = 256,
    energy_percentile: float = 50,     # Keep frames above this percentile of energy
    min_segment_ms: float = 50,        # Minimum segment length to keep
    max_gap_ms: float = 200,           # Maximum gap to bridge between segments  
    fade_ms: float = 10                # Fade in/out to avoid clicks
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter audio to keep only frames where speaker is clearly dominant.
    
    Uses global energy percentile approach:
    - Compute energy for all frames
    - Keep frames above a percentile threshold (main speaker is louder)
    - Clean up with segment merging and gap filling
    
    Returns:
        filtered_audio: Audio with non-dominant parts zeroed out
        mask: Boolean mask of kept frames
    """
    # Compute RMS energy per frame
    energy = compute_frame_energy(audio, frame_length, hop_length)
    
    # Convert to dB for thresholding
    energy_db = 20 * np.log10(energy + 1e-10)
    
    # Compute adaptive threshold based on percentile
    threshold_energy = compute_global_energy_percentile(energy, energy_percentile)
    threshold_db = 20 * np.log10(threshold_energy + 1e-10)
    
    print(f"  Energy threshold (p{energy_percentile}): {threshold_db:.1f} dB")
    print(f"  Energy range: {np.min(energy_db):.1f} to {np.max(energy_db):.1f} dB")
    
    # Create mask: keep frames above threshold
    mask = energy >= threshold_energy
    
    print(f"  Initial mask: {np.sum(mask)}/{len(mask)} frames ({100*np.mean(mask):.1f}%)")
    
    # Remove short segments (likely noise bursts)
    min_frames = max(1, int(min_segment_ms * sr / (1000 * hop_length)))
    mask = remove_short_segments(mask, min_frames)
    
    print(f"  After removing short segments: {np.sum(mask)} frames")
    
    # Fill short gaps (avoid choppy audio)
    max_gap_frames = int(max_gap_ms * sr / (1000 * hop_length))
    mask = fill_short_gaps(mask, max_gap_frames)
    
    print(f"  After filling gaps: {np.sum(mask)} frames")
    
    # Expand mask to sample level
    sample_mask = frames_to_samples(mask, hop_length, len(audio))
    
    # Apply fade to avoid clicks
    fade_samples = int(fade_ms * sr / 1000)
    sample_mask = apply_fade(sample_mask, fade_samples)
    
    # Apply mask
    filtered_audio = audio * sample_mask
    
    return filtered_audio, mask


def remove_short_segments(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Remove True segments shorter than min_length."""
    mask = mask.copy()
    
    in_segment = False
    segment_start = 0
    
    for i, val in enumerate(mask):
        if val and not in_segment:
            in_segment = True
            segment_start = i
        elif not val and in_segment:
            in_segment = False
            if i - segment_start < min_length:
                mask[segment_start:i] = False
    
    # Handle segment at end
    if in_segment and len(mask) - segment_start < min_length:
        mask[segment_start:] = False
    
    return mask


def fill_short_gaps(mask: np.ndarray, min_gap: int) -> np.ndarray:
    """Fill False gaps shorter than min_gap."""
    mask = mask.copy()
    
    in_gap = False
    gap_start = 0
    
    for i, val in enumerate(mask):
        if not val and not in_gap:
            in_gap = True
            gap_start = i
        elif val and in_gap:
            in_gap = False
            if i - gap_start < min_gap:
                mask[gap_start:i] = True
    
    return mask


def frames_to_samples(
    frame_mask: np.ndarray,
    hop_length: int,
    num_samples: int
) -> np.ndarray:
    """Expand frame-level mask to sample-level."""
    sample_mask = np.zeros(num_samples, dtype=np.float32)
    
    for i, val in enumerate(frame_mask):
        if val:
            start = i * hop_length
            end = min(start + hop_length, num_samples)
            sample_mask[start:end] = 1.0
    
    return sample_mask


def apply_fade(mask: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply fade in/out at transitions to avoid clicks."""
    mask = mask.copy().astype(np.float32)
    
    # Find transitions
    diff = np.diff(mask, prepend=0)
    fade_in_points = np.where(diff > 0)[0]
    fade_out_points = np.where(diff < 0)[0]
    
    # Apply fade in
    for point in fade_in_points:
        start = max(0, point - fade_samples)
        fade = np.linspace(0, 1, point - start)
        mask[start:point] = fade
    
    # Apply fade out
    for point in fade_out_points:
        end = min(len(mask), point + fade_samples)
        fade = np.linspace(1, 0, end - point)
        mask[point:end] = fade
    
    return mask


# ============ For your callbot pipeline ============

def process_for_stt(
    audio: np.ndarray,
    sr: int = 16000,
    aggressive: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Process audio for STT by removing background voices.
    
    Args:
        audio: Input audio
        sr: Sample rate
        aggressive: If True, use stricter filtering (keeps less audio)
    
    Returns:
        filtered_audio: Cleaned audio
        stats: Processing statistics
    """
    params = {
        'energy_percentile': 60 if aggressive else 40,  # Higher = keep less
        'min_segment_ms': 100 if aggressive else 50,
        'max_gap_ms': 150 if aggressive else 250,
    }
    
    filtered, mask = filter_dominant_speaker(audio, sr, **params)
    
    # Stats for debugging
    stats = {
        'kept_ratio': np.mean(mask),
        'original_rms': np.sqrt(np.mean(audio ** 2)),
        'filtered_rms': np.sqrt(np.mean(filtered ** 2)),
        'num_segments': count_segments(mask),
    }
    
    return filtered, stats


def count_segments(mask: np.ndarray) -> int:
    """Count number of True segments in mask."""
    diff = np.diff(mask.astype(int), prepend=0)
    return np.sum(diff > 0)


# ============ Usage Example ============

if __name__ == "__main__":
    import os
    
    # Use the same output from test_denoise.py
    input_file = "output/step1_left_channel.wav"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run test_denoise.py first.")
        exit(1)
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=16000)
    print(f"Loaded {input_file}: {len(audio)/sr:.2f}s at {sr}Hz")
    
    # Process with normal mode
    print("\n--- Normal Mode ---")
    filtered, stats = process_for_stt(audio, sr, aggressive=False)
    
    print(f"\nResults:")
    print(f"  Kept {stats['kept_ratio']*100:.1f}% of audio")
    print(f"  Found {stats['num_segments']} speech segments")
    print(f"  Original RMS: {stats['original_rms']:.4f}")
    print(f"  Filtered RMS: {stats['filtered_rms']:.4f}")
    
    # Save
    sf.write("output/filtered.wav", filtered, sr)
    print(f"  Saved to: output/filtered.wav")
    
    # Also try aggressive mode for comparison
    print("\n--- Aggressive Mode ---")
    filtered_agg, stats_agg = process_for_stt(audio, sr, aggressive=True)
    
    print(f"\nResults:")
    print(f"  Kept {stats_agg['kept_ratio']*100:.1f}% of audio")
    print(f"  Found {stats_agg['num_segments']} speech segments")
    
    sf.write("output/filtered_aggressive.wav", filtered_agg, sr)
    print(f"  Saved to: output/filtered_aggressive.wav")