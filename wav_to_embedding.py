import os
import time
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F

# Workaround for torchaudio 2.5+ compatibility with speechbrain
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'

# Workaround for Windows symlink issues - set environment variable before import
os.environ["SB_LOCAL_STRATEGY"] = "copy"

from speechbrain.inference.speaker import EncoderClassifier


def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


def cluster_speakers(embeddings, wav_files, lengths, similarity_threshold=0.4):
    """
    Cluster VAD segments by speaker based on embedding similarity.
    Identify main speaker (most speaking time) vs others.
    
    Args:
        embeddings: Tensor of shape [n, 1, 192]
        wav_files: List of wav filenames
        lengths: List of audio lengths in samples
        similarity_threshold: Threshold for same speaker (default 0.4)
    
    Returns:
        speaker_labels: Dict mapping filename to speaker ID
        main_speaker_id: ID of the main speaker
    """
    n = len(wav_files)
    
    # Build similarity matrix
    similarity_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    
    # Simple clustering: group segments by similarity
    speaker_labels = [-1] * n
    current_speaker = 0
    
    for i in range(n):
        if speaker_labels[i] == -1:
            # Start new speaker cluster
            speaker_labels[i] = current_speaker
            
            # Find all similar segments
            for j in range(i + 1, n):
                if speaker_labels[j] == -1:
                    if similarity_matrix[i, j] > similarity_threshold:
                        speaker_labels[j] = current_speaker
            
            current_speaker += 1
    
    # Calculate total speaking time per speaker
    speaker_times = {}
    for i, label in enumerate(speaker_labels):
        if label not in speaker_times:
            speaker_times[label] = 0
        speaker_times[label] += lengths[i]
    
    # Main speaker = most speaking time
    main_speaker_id = max(speaker_times, key=speaker_times.get)
    
    return speaker_labels, main_speaker_id, speaker_times, similarity_matrix


def main():
    # Load the pre-trained ECAPA-TDNN model from local folder
    print("Loading ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="spkrec-ecapa-voxceleb",
        savedir="spkrec-ecapa-voxceleb"
    )
    print("Model loaded.")
    
    # Directory containing wav files
    chunks_dir = "output/chunks"
    
    # Get all wav files
    wav_files = [f for f in os.listdir(chunks_dir) if f.endswith('.wav')]
    wav_files.sort()
    
    print(f"\nFound {len(wav_files)} wav files in {chunks_dir}")
    print("-" * 60)
    
    # Load all audio files first
    signals = []
    lengths = []
    print("Loading audio files...")
    for wav_file in wav_files:
        wav_path = os.path.join(chunks_dir, wav_file)
        data, fs = sf.read(wav_path)
        signal = torch.tensor(data).float()
        signals.append(signal)
        lengths.append(len(signal))
        print(f"  {wav_file}: {len(signal)} samples")
    
    # Pad all signals to the same length for batching
    max_len = max(lengths)
    batch_signals = []
    for signal in signals:
        if len(signal) < max_len:
            padded = torch.zeros(max_len)
            padded[:len(signal)] = signal
            batch_signals.append(padded)
        else:
            batch_signals.append(signal)
    
    # Stack into batch tensor [batch_size, max_length]
    batch = torch.stack(batch_signals)
    wav_lens = torch.tensor(lengths).float() / max_len  # Relative lengths
    
    print(f"\nBatch shape: {batch.shape}")
    print("-" * 60)
    
    # Time only the embedding extraction
    start_time = time.time()
    embeddings = classifier.encode_batch(batch, wav_lens)
    elapsed = time.time() - start_time
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print("-" * 60)
    print(f"Total files processed: {len(wav_files)}")
    print(f"Time for batch embedding: {elapsed:.4f}s")
    print(f"Average time per file: {elapsed / len(wav_files):.4f}s")
    
    # Cluster speakers and detect main voice
    print("\n" + "=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)
    
    speaker_labels, main_speaker_id, speaker_times, similarity_matrix = cluster_speakers(
        embeddings, wav_files, lengths, similarity_threshold=0.4
    )
    
    # Print similarity matrix
    print("\nSimilarity Matrix:")
    short_names = [f.replace("step2_gated_chunk_", "").replace(".wav", "") for f in wav_files]
    
    print(f"\n{'':>12}", end="")
    for name in short_names:
        print(f"{name[:8]:>10}", end="")
    print()
    
    for i, name in enumerate(short_names):
        print(f"{name[:12]:>12}", end="")
        for j in range(len(wav_files)):
            sim = similarity_matrix[i, j].item()
            print(f"{sim:>10.4f}", end="")
        print()
    
    # Print speaker assignments
    print("\n" + "-" * 60)
    print("SPEAKER ASSIGNMENTS:")
    print("-" * 60)
    
    sample_rate = 8000  # Assuming 8kHz sample rate
    for i, wav_file in enumerate(wav_files):
        speaker_id = speaker_labels[i]
        duration_ms = lengths[i] / sample_rate * 1000
        is_main = "** MAIN **" if speaker_id == main_speaker_id else ""
        print(f"  Speaker {speaker_id}: {wav_file} ({duration_ms:.0f}ms) {is_main}")
    
    # Print speaker summary
    print("\n" + "-" * 60)
    print("SPEAKER SUMMARY:")
    print("-" * 60)
    
    num_speakers = len(speaker_times)
    print(f"Total speakers detected: {num_speakers}")
    
    for speaker_id in sorted(speaker_times.keys()):
        total_samples = speaker_times[speaker_id]
        total_ms = total_samples / sample_rate * 1000
        is_main = "<-- MAIN VOICE" if speaker_id == main_speaker_id else ""
        segments = [wav_files[i] for i, s in enumerate(speaker_labels) if s == speaker_id]
        print(f"\n  Speaker {speaker_id}: {total_ms:.0f}ms total ({len(segments)} segments) {is_main}")
        for seg in segments:
            print(f"    - {seg}")
    
    # Return results
    result = {
        "embeddings": embeddings,
        "wav_files": wav_files,
        "speaker_labels": speaker_labels,
        "main_speaker_id": main_speaker_id,
        "speaker_times": speaker_times,
        "similarity_matrix": similarity_matrix
    }
    
    return result

if __name__ == "__main__":
    result = main()
