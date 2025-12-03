"""
Script to prepare training data from labels_v1.csv.

This script reads the labeled segments from labels_v1.csv and extracts
audio segments into a directory structure suitable for training:

output_dir/
    speech/
        segment_001.wav
        segment_002.wav
        ...
    noise/
        segment_001.wav
        segment_002.wav
        ...
"""

import os
import pandas as pd
from pydub import AudioSegment
from pathlib import Path


def find_audio_file(filename: str, search_dirs: list) -> str | None:
    """
    Find an audio file in the given search directories.
    
    Args:
        filename: Name of the audio file to find
        search_dirs: List of directories to search in
        
    Returns:
        Full path to the file if found, None otherwise
    """
    for search_dir in search_dirs:
        path = os.path.join(search_dir, filename)
        if os.path.exists(path):
            return path
    return None


def extract_segment(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str
) -> bool:
    """
    Extract a segment from an audio file and save it.
    
    Args:
        audio_path: Path to the source audio file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        output_path: Path to save the extracted segment
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio based on file extension
        ext = Path(audio_path).suffix.lower()
        if ext == '.mp3':
            audio = AudioSegment.from_mp3(audio_path)
        elif ext == '.wav':
            audio = AudioSegment.from_wav(audio_path)
        elif ext == '.m4a':
            audio = AudioSegment.from_file(audio_path, format='m4a')
        else:
            audio = AudioSegment.from_file(audio_path)
        
        # Extract left channel only if stereo
        if audio.channels == 2:
            audio = audio.split_to_mono()[0]
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        # Export as wav
        segment.export(output_path, format='wav')
        return True
    except Exception as e:
        print(f"Error extracting segment from {audio_path}: {e}")
        return False


def prepare_training_data(
    csv_path: str,
    output_dir: str,
    search_dirs: list,
    min_duration_ms: int = 300,
    max_duration_ms: int = 10000
):
    """
    Prepare training data from the labels CSV file.
    
    Args:
        csv_path: Path to the labels_v1.csv file
        output_dir: Output directory for the training data
        search_dirs: List of directories to search for source audio files
        min_duration_ms: Minimum segment duration to include (default 300ms)
        max_duration_ms: Maximum segment duration to include (default 10s)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled segments from {csv_path}")
    
    # Show label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Create output directories
    speech_dir = os.path.join(output_dir, "speech")
    noise_dir = os.path.join(output_dir, "noise")
    os.makedirs(speech_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        'speech_extracted': 0,
        'noise_extracted': 0,
        'skipped_not_found': 0,
        'skipped_too_short': 0,
        'skipped_too_long': 0,
        'failed': 0
    }
    
    # Cache for loaded audio files (to avoid reloading same file)
    audio_cache = {}
    
    # Process each segment
    for idx, row in df.iterrows():
        filename = row['filename']
        start_ms = int(row['segment_start_ms'])
        end_ms = int(row['segment_end_ms'])
        duration_ms = int(row['duration_ms'])
        label = row['label']
        
        # Check duration constraints
        if duration_ms < min_duration_ms:
            stats['skipped_too_short'] += 1
            continue
        if duration_ms > max_duration_ms:
            stats['skipped_too_long'] += 1
            continue
        
        # Find the source audio file
        audio_path = find_audio_file(filename, search_dirs)
        if audio_path is None:
            stats['skipped_not_found'] += 1
            continue
        
        # Determine output directory and filename
        if label == 'noise':
            out_dir = noise_dir
            segment_idx = stats['noise_extracted']
        else:  # 'not_noise' -> speech
            out_dir = speech_dir
            segment_idx = stats['speech_extracted']
        
        # Create output filename with source info
        base_name = Path(filename).stem
        output_filename = f"{base_name}_{start_ms}_{end_ms}.wav"
        output_path = os.path.join(out_dir, output_filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            if label == 'noise':
                stats['noise_extracted'] += 1
            else:
                stats['speech_extracted'] += 1
            continue
        
        # Extract and save segment
        success = extract_segment(audio_path, start_ms, end_ms, output_path)
        
        if success:
            if label == 'noise':
                stats['noise_extracted'] += 1
            else:
                stats['speech_extracted'] += 1
        else:
            stats['failed'] += 1
        
        # Progress update every 100 segments
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} segments...")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")
    print(f"Speech segments extracted: {stats['speech_extracted']}")
    print(f"Noise segments extracted:  {stats['noise_extracted']}")
    print(f"Skipped (file not found):  {stats['skipped_not_found']}")
    print(f"Skipped (too short):       {stats['skipped_too_short']}")
    print(f"Skipped (too long):        {stats['skipped_too_long']}")
    print(f"Failed extractions:        {stats['failed']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  speech/: {stats['speech_extracted']} files")
    print(f"  noise/:  {stats['noise_extracted']} files")
    
    return stats


def main():
    # Configuration
    CSV_PATH = "denoise_data/labels_v1.csv"
    OUTPUT_DIR = "denoise_data/training_data"
    
    # Directories to search for source audio files
    SEARCH_DIRS = [
        "denoise_data/samples",
        "denoise_data/samples_uat_0411"
    ]
    
    print("="*60)
    print("Preparing Training Data from labels_v1.csv")
    print("="*60)
    
    # Run extraction
    stats = prepare_training_data(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        search_dirs=SEARCH_DIRS,
        min_duration_ms=300,   # Minimum 300ms
        max_duration_ms=10000  # Maximum 10 seconds
    )
    
    print("\nDone! You can now use the training data with train_wav2vec2_classifier.py")
    print(f"Set DATA_DIR = '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
