import os
import numpy as np
import librosa
import soundfile as sf
import torch
from pydub import AudioSegment

INPUT_FILE = r'D:\callbot-denoiser\denoise_data\call_id_mass_20251128\1a1afc59-28d1-404b-934e-0bf7aecda8dc.mp3'
OUTPUT_DIR = r'D:\callbot-denoiser\output\vad_analysis'


def load_silero_vad():
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


def extract_left_channel(audio_path):
    """Extract left channel from stereo audio, return audio array and sample rate."""
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    if audio.ndim == 1:
        return audio, sr
    else:
        return audio[0], sr


def detect_vads(audio, sr, model, get_speech_timestamps, 
                threshold=0.3, min_speech_duration_ms=100, 
                min_silence_duration_ms=100, speech_pad_ms=30):
    """Detect VAD segments."""
    # Resample to 16kHz for VAD
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio
    
    waveform_tensor = torch.FloatTensor(audio_16k)
    
    vad_results = get_speech_timestamps(
        waveform_tensor,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    return vad_results


def compute_voicing_ratio(segment, sr):
    """Compute ratio of voiced frames (frames with detectable pitch)."""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            segment, fmin=50, fmax=400, sr=sr
        )
        # Count frames with valid pitch (not NaN)
        valid_f0 = ~np.isnan(f0)
        if len(f0) == 0:
            return 0.0
        return np.mean(valid_f0)  # Ratio of voiced frames
    except:
        return 0.0


def compute_harmonic_clarity(segment, sr):
    """Compute harmonic clarity - ratio of harmonic energy to total energy."""
    try:
        if len(segment) < 2048:
            return 0.0
        harmonic, percussive = librosa.effects.hpss(segment)
        harmonic_energy = np.sum(harmonic ** 2)
        total_energy = np.sum(segment ** 2) + 1e-10
        clarity = harmonic_energy / total_energy
        return min(clarity, 1.0)  # Cap at 1.0
    except:
        return 0.0


def compute_vad_stats(audio, sr, vad_results):
    """Compute statistics for each VAD segment."""
    stats = []
    
    for i, vad in enumerate(vad_results):
        # Convert from 16kHz samples to original sample rate
        start_sample = int(vad['start'] / 16000 * sr)
        end_sample = int(vad['end'] / 16000 * sr)
        
        segment = audio[start_sample:end_sample]
        
        if len(segment) == 0:
            continue
        
        # Time info
        start_sec = start_sample / sr
        end_sec = end_sample / sr
        duration = end_sec - start_sec
        
        # Peak dB
        peak = np.max(np.abs(segment))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Harmonic Clarity (clean voice has high harmonic content)
        harmonic_clarity = compute_harmonic_clarity(segment, sr)
        
        # Voicing Ratio (percentage of voiced frames)
        voicing_ratio = compute_voicing_ratio(segment, sr)
        
        stats.append({
            'idx': i + 1,
            'start': start_sec,
            'end': end_sec,
            'duration': duration,
            'peak_db': peak_db,
            'harmonic_clarity': harmonic_clarity,
            'voicing_ratio': voicing_ratio,
            'start_sample': start_sample,
            'end_sample': end_sample
        })
    
    return stats


def save_vad_chunks(audio, sr, stats, output_dir):
    """Save each VAD segment as a separate WAV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    for s in stats:
        segment = audio[s['start_sample']:s['end_sample']]
        filename = f"vad_{s['idx']:03d}_{s['start']:.2f}s_{s['duration']:.2f}s.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, segment, sr)
        s['file'] = filepath


def print_stats_table(stats):
    """Print statistics table."""
    print("\n" + "=" * 85)
    print(f"{'#':>3} | {'Start':>7} | {'End':>7} | {'Dur':>5} | {'Peak':>7} | {'Harmonic':>8} | {'Voicing':>7}")
    print(f"{'':>3} | {'':>7} | {'':>7} | {'(s)':>5} | {'(dB)':>7} | {'Clarity':>8} | {'Ratio':>7}")
    print("-" * 85)
    
    for s in stats:
        print(f"{s['idx']:>3} | {s['start']:>7.2f} | {s['end']:>7.2f} | {s['duration']:>5.2f} | "
              f"{s['peak_db']:>7.1f} | {s['harmonic_clarity']:>8.2f} | {s['voicing_ratio']:>7.2f}")
    
    print("=" * 85)


def analyze_dominance(stats):
    """Analyze which segments are likely the dominant speaker."""
    if not stats:
        return
    
    # Extract metrics
    peak_values = np.array([s['peak_db'] for s in stats])
    harmonic_values = np.array([s['harmonic_clarity'] for s in stats])
    voicing_values = np.array([s['voicing_ratio'] for s in stats])
    durations = np.array([s['duration'] for s in stats])
    
    # Compute percentiles
    peak_p75 = np.percentile(peak_values, 75)
    peak_p50 = np.percentile(peak_values, 50)
    peak_p25 = np.percentile(peak_values, 25)
    
    print("\n" + "=" * 70)
    print("DOMINANCE ANALYSIS")
    print("=" * 70)
    
    print(f"\nPeak dB Distribution:")
    print(f"  P25: {peak_p25:.1f} dB | P50: {peak_p50:.1f} dB | P75: {peak_p75:.1f} dB")
    print(f"  Range: {np.min(peak_values):.1f} to {np.max(peak_values):.1f} dB")
    
    print(f"\nHarmonic Clarity Distribution:")
    print(f"  Mean: {np.mean(harmonic_values):.2f} | Median: {np.median(harmonic_values):.2f}")
    print(f"  Range: {np.min(harmonic_values):.2f} to {np.max(harmonic_values):.2f}")
    
    print(f"\nVoicing Ratio Distribution:")
    print(f"  Mean: {np.mean(voicing_values):.2f} | Median: {np.median(voicing_values):.2f}")
    print(f"  Range: {np.min(voicing_values):.2f} to {np.max(voicing_values):.2f}")
    
    print(f"\nDuration: Total speech = {np.sum(durations):.1f}s")
    
    # Classify by peak dB
    loud_segments = [s for s in stats if s['peak_db'] >= peak_p75]
    quiet_segments = [s for s in stats if s['peak_db'] < peak_p25]
    
    print(f"\nSegment Classification (by Peak dB):")
    print(f"  LOUD (>= {peak_p75:.1f} dB): {len(loud_segments)} segments, "
          f"{sum(s['duration'] for s in loud_segments):.1f}s total")
    print(f"  QUIET (< {peak_p25:.1f} dB): {len(quiet_segments)} segments, "
          f"{sum(s['duration'] for s in quiet_segments):.1f}s total")
    
    # Compute dominance score: high peak + high harmonic clarity + high voicing ratio
    for s in stats:
        # Normalize each feature to 0-1 range
        peak_norm = (s['peak_db'] - np.min(peak_values)) / (np.max(peak_values) - np.min(peak_values) + 1e-10)
        harmonic_norm = s['harmonic_clarity']  # Already 0-1
        voicing_norm = s['voicing_ratio']  # Already 0-1
        
        # Weighted score
        s['dominance_score'] = 0.4 * peak_norm + 0.3 * harmonic_norm + 0.3 * voicing_norm
    
    # Top 10 by dominance score
    print("\n" + "-" * 70)
    print("TOP 10 DOMINANT SEGMENTS (likely main speaker):")
    print("-" * 70)
    sorted_by_score = sorted(stats, key=lambda x: x['dominance_score'], reverse=True)[:10]
    for s in sorted_by_score:
        print(f"  #{s['idx']:03d}: {s['start']:.2f}s - {s['end']:.2f}s | "
              f"Score: {s['dominance_score']:.2f} | Peak: {s['peak_db']:.0f}dB | "
              f"Harmonic: {s['harmonic_clarity']:.2f} | Voicing: {s['voicing_ratio']:.2f}")
    
    # Top 10 likely background/noise
    print("\n" + "-" * 70)
    print("TOP 10 BACKGROUND/NOISE SEGMENTS (likely crosstalk):")
    print("-" * 70)
    sorted_by_score_asc = sorted(stats, key=lambda x: x['dominance_score'])[:10]
    for s in sorted_by_score_asc:
        print(f"  #{s['idx']:03d}: {s['start']:.2f}s - {s['end']:.2f}s | "
              f"Score: {s['dominance_score']:.2f} | Peak: {s['peak_db']:.0f}dB | "
              f"Harmonic: {s['harmonic_clarity']:.2f} | Voicing: {s['voicing_ratio']:.2f}")
    
    return {
        'loud_segments': loud_segments,
        'quiet_segments': quiet_segments,
        'peak_p75': peak_p75,
        'peak_p50': peak_p50,
        'peak_p25': peak_p25
    }


def main():
    print("=" * 60)
    print("VAD Extraction & Statistics Analysis")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Step 1: Extract left channel
    print("\n[1] Extracting left channel...")
    audio, sr = extract_left_channel(INPUT_FILE)
    print(f"    Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")
    
    # Step 2: Load VAD model
    print("\n[2] Loading VAD model...")
    model, get_speech_timestamps = load_silero_vad()
    
    # Step 3: Detect VADs
    print("\n[3] Detecting voice activity...")
    vad_results = detect_vads(audio, sr, model, get_speech_timestamps)
    print(f"    Found {len(vad_results)} speech segments")
    
    # Step 4: Compute statistics
    print("\n[4] Computing statistics...")
    stats = compute_vad_stats(audio, sr, vad_results)
    
    # Step 5: Save chunks
    print("\n[5] Saving VAD chunks...")
    save_vad_chunks(audio, sr, stats, OUTPUT_DIR)
    print(f"    Saved {len(stats)} files to {OUTPUT_DIR}")
    
    # Step 6: Print results
    print_stats_table(stats)
    
    # Step 7: Analyze dominance
    analysis = analyze_dominance(stats)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return stats, analysis


if __name__ == "__main__":
    stats, analysis = main()
