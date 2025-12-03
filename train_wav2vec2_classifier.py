"""
Example script demonstrating how to train and use the Wav2Vec2 Speech/Noise Classifier.

This script shows:
1. How to prepare a dataset
2. How to train the model
3. How to evaluate and use the model for inference
"""

import os
import torch
from wav2vec2_classifier import (
    Wav2Vec2MLPClassifier,
    train_model,
    save_model,
    load_model_weights,
    classify_audio,
    load_audio
)


def prepare_sample_dataset(data_dir: str):
    """
    Prepare a dataset from a directory structure.
    
    Expected structure:
    data_dir/
        speech/
            audio1.wav
            audio2.mp3
            ...
        noise/
            noise1.wav
            noise2.mp3
            ...
    
    Returns:
        Tuple of (file_paths, labels)
    """
    file_paths = []
    labels = []
    
    # Load speech files (label = 1)
    speech_dir = os.path.join(data_dir, "speech")
    if os.path.exists(speech_dir):
        for f in os.listdir(speech_dir):
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_paths.append(os.path.join(speech_dir, f))
                labels.append(1)
        print(f"Found {labels.count(1)} speech files")
    
    # Load noise files (label = 0)
    noise_dir = os.path.join(data_dir, "noise")
    if os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_paths.append(os.path.join(noise_dir, f))
                labels.append(0)
        print(f"Found {labels.count(0)} noise files")
    
    return file_paths, labels


def main():
    """Main training and evaluation example."""
    
    print("=" * 60)
    print("Wav2Vec2 + MLP Speech/Noise Classifier - Training Example")
    print("=" * 60)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # =========================================================================
    # Option 1: Train with real data
    # =========================================================================
    # Uncomment and modify this section if you have labeled data
    
    # DATA_DIR = "path/to/your/data"  # Directory with speech/ and noise/ subdirs
    # 
    # # Prepare dataset
    # file_paths, labels = prepare_sample_dataset(DATA_DIR)
    # 
    # # Split into train/val (80/20)
    # from sklearn.model_selection import train_test_split
    # train_files, val_files, train_labels, val_labels = train_test_split(
    #     file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    # )
    # 
    # # Create and train model
    # model = Wav2Vec2MLPClassifier(freeze_wav2vec2=True)
    # 
    # history = train_model(
    #     model=model,
    #     train_files=train_files,
    #     train_labels=train_labels,
    #     val_files=val_files,
    #     val_labels=val_labels,
    #     epochs=10,
    #     batch_size=4,
    #     learning_rate=1e-4,
    #     device=device
    # )
    # 
    # # Save trained model
    # save_model(model, "speech_noise_classifier.pt")
    
    # =========================================================================
    # Option 2: Demo with synthetic data (no real audio files needed)
    # =========================================================================
    print("\n--- Running demo with synthetic data ---\n")
    
    # Create model
    print("Loading Wav2Vec2 model (this may take a moment)...")
    model = Wav2Vec2MLPClassifier(
        wav2vec2_model_name="facebook/wav2vec2-base",
        hidden_dim=256,
        dropout=0.3,
        freeze_wav2vec2=True
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create synthetic training data
    print("\nCreating synthetic training data...")
    import numpy as np
    
    def create_synthetic_noise(duration_sec=2, sample_rate=16000):
        """Create white noise."""
        return torch.randn(duration_sec * sample_rate) * 0.1
    
    def create_synthetic_speech(duration_sec=2, sample_rate=16000):
        """Create speech-like signal (multiple harmonics)."""
        t = torch.linspace(0, duration_sec, duration_sec * sample_rate)
        # Simulate speech with fundamental + harmonics
        signal = torch.zeros_like(t)
        for freq in [100, 200, 300, 400]:  # Harmonics
            signal += torch.sin(2 * np.pi * freq * t) * (1 / (freq / 100))
        # Add some amplitude modulation
        envelope = torch.sin(2 * np.pi * 3 * t) * 0.3 + 0.7
        return signal * envelope * 0.1
    
    # Generate synthetic dataset
    n_samples = 20  # Small for demo
    synthetic_waveforms = []
    synthetic_labels = []
    
    for _ in range(n_samples // 2):
        synthetic_waveforms.append(create_synthetic_noise())
        synthetic_labels.append(0)  # noise
    
    for _ in range(n_samples // 2):
        synthetic_waveforms.append(create_synthetic_speech())
        synthetic_labels.append(1)  # speech
    
    # Simple training loop with synthetic data
    print(f"Training on {n_samples} synthetic samples...")
    
    model.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(3):
        total_loss = 0
        correct = 0
        
        for waveform, label in zip(synthetic_waveforms, synthetic_labels):
            waveform = waveform.to(device)
            label_tensor = torch.tensor([label], device=device)
            
            optimizer.zero_grad()
            logits = model(waveform, 16000)
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1).item()
            correct += (pred == label)
        
        acc = correct / n_samples
        print(f"Epoch {epoch+1}/3 - Loss: {total_loss/n_samples:.4f}, Acc: {acc:.2%}")
    
    # Test inference
    print("\n--- Testing inference ---\n")
    
    model.eval()
    
    # Test on noise
    test_noise = create_synthetic_noise(3).to(device)
    pred_class, confidence = model.predict(test_noise, 16000)
    print(f"Synthetic noise -> Predicted: {'speech' if pred_class == 1 else 'noise'} "
          f"(confidence: {confidence:.2%})")
    
    # Test on speech-like
    test_speech = create_synthetic_speech(3).to(device)
    pred_class, confidence = model.predict(test_speech, 16000)
    print(f"Synthetic speech -> Predicted: {'speech' if pred_class == 1 else 'noise'} "
          f"(confidence: {confidence:.2%})")
    
    # =========================================================================
    # Save and load example
    # =========================================================================
    print("\n--- Save/Load example ---\n")
    
    model_path = "speech_noise_classifier_demo.pt"
    save_model(model, model_path)
    
    # Load into new model
    new_model = Wav2Vec2MLPClassifier()
    load_model_weights(new_model, model_path)
    
    # Clean up demo model file
    os.remove(model_path)
    print("Cleaned up demo model file")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo train on real data:")
    print("1. Create a directory with 'speech/' and 'noise/' subdirectories")
    print("2. Place labeled audio files in each subdirectory")
    print("3. Modify this script to use your data directory")
    print("4. Run training with train_model()")


if __name__ == "__main__":
    main()
