import os
from vad import (
    split_stereo_channels_fn,
    detect_speech_vads_fn,
    aggregate_vads_fn
)


def process_audio(input_path: str, output_folder: str) -> dict:
    """
    Process a stereo audio file: split channels, detect VADs, and aggregate results.
    
    Args:
        input_path: Path to the input stereo WAV file
        output_folder: Folder to save all output files
    
    Returns:
        dict: Aggregated VAD results with role labels
    """
    print(f"Processing: {input_path}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Split stereo channels
    print("\n[Step 1] Splitting stereo channels...")
    split_result = split_stereo_channels_fn(input_path, output_folder)
    left_path = split_result['left_path']
    right_path = split_result['right_path']
    print(f"  Left channel: {left_path}")
    print(f"  Right channel: {right_path}")
    
    # Step 2: Detect VADs in left channel (agent)
    print("\n[Step 2] Detecting VADs in left channel (agent)...")
    left_chunks_folder = os.path.join(output_folder, "left_channel_chunks")
    left_vads_result = detect_speech_vads_fn(
        audio_channel_path=left_path,
        output_folder=left_chunks_folder
    )
    left_vads = left_vads_result['vads']
    print(f"  Found {len(left_vads)} speech chunks in left channel")
    
    # Step 3: Detect VADs in right channel (customer)
    print("\n[Step 3] Detecting VADs in right channel (customer)...")
    right_chunks_folder = os.path.join(output_folder, "right_channel_chunks")
    right_vads_result = detect_speech_vads_fn(
        audio_channel_path=right_path,
        output_folder=right_chunks_folder
    )
    right_vads = right_vads_result['vads']
    print(f"  Found {len(right_vads)} speech chunks in right channel")
    
    # Step 4: Aggregate VADs
    print("\n[Step 4] Aggregating VADs...")
    aggregated_result = aggregate_vads_fn(left_vads, right_vads)
    transcribed_vads = aggregated_result['transcribed_vads']
    print(f"  Total aggregated chunks: {len(transcribed_vads)}")
    
    # Print aggregated VADs
    print("\n[Results] Aggregated VADs:")
    for i, vad in enumerate(transcribed_vads):
        print(f"  {i+1}. [{vad['role']:8}] {vad['start']:.2f}s - {vad['end']:.2f}s (duration: {vad['duration']:.2f}s)")
    
    return aggregated_result


def main():
    # Input and output configuration
    input_path = "11a72914-ebfa-4115-bd23-0b968a88c837_denoised.wav"
    output_folder = "preprocess_data"
    
    # Process the audio
    result = process_audio(input_path, output_folder)
    
    print(f"\nProcessing complete! Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
