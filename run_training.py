"""
Script to train Wav2Vec2 classifier on training_data/
"""

import os
import random
import torch
from sklearn.model_selection import train_test_split
from wav2vec2_classifier import (
    Wav2Vec2MLPClassifier,
    train_model,
    save_model,
)


def main():
    print('=' * 60)
    print('Training Wav2Vec2 Classifier on training_data/')
    print('=' * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    DATA_DIR = 'denoise_data/training_data'

    # Prepare dataset - collect files by class
    speech_files = []
    noise_files = []

    speech_dir = os.path.join(DATA_DIR, 'speech')
    if os.path.exists(speech_dir):
        for f in os.listdir(speech_dir):
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                speech_files.append(os.path.join(speech_dir, f))
        print(f'Found {len(speech_files)} speech files')

    noise_dir = os.path.join(DATA_DIR, 'noise')
    if os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                noise_files.append(os.path.join(noise_dir, f))
        print(f'Found {len(noise_files)} noise files')

    # Downsample majority class to balance dataset
    min_samples = min(len(speech_files), len(noise_files))
    print(f'Balancing dataset: downsampling to {min_samples} samples per class')
    
    random.seed(42)
    if len(speech_files) > min_samples:
        speech_files = random.sample(speech_files, min_samples)
    if len(noise_files) > min_samples:
        noise_files = random.sample(noise_files, min_samples)
    
    # Combine balanced dataset
    file_paths = speech_files + noise_files
    labels = [1] * len(speech_files) + [0] * len(noise_files)
    
    print(f'Balanced dataset: {len(speech_files)} speech + {len(noise_files)} noise = {len(file_paths)} total')

    # Split into train/val (80/20) with stratification
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    # Create and train model
    print('Loading Wav2Vec2 model...')
    # model = Wav2Vec2MLPClassifier(freeze_wav2vec2=True)
    model = Wav2Vec2MLPClassifier(
        freeze_wav2vec2=True, 
        wav2vec2_model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        hidden_dim=256,
        dropout=0.2
    )
    print('Model loaded!')

    history = train_model(
        model=model,
        train_files=train_files,
        train_labels=train_labels,
        val_files=val_files,
        val_labels=val_labels,
        epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        device=device
    )

    # Save trained model
    save_model(model, 'speech_noise_vi_classifier.pt')
    print('Training complete! Model saved to speech_noise_classifier.pt')


if __name__ == '__main__':
    main()
