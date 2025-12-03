"""
Script to train Wav2Vec2 classifier on training_data/
"""

import os
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

    # Prepare dataset
    file_paths = []
    labels = []

    speech_dir = os.path.join(DATA_DIR, 'speech')
    if os.path.exists(speech_dir):
        for f in os.listdir(speech_dir):
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_paths.append(os.path.join(speech_dir, f))
                labels.append(1)
        print(f'Found {labels.count(1)} speech files')

    noise_dir = os.path.join(DATA_DIR, 'noise')
    if os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_paths.append(os.path.join(noise_dir, f))
                labels.append(0)
        print(f'Found {labels.count(0)} noise files')

    print(f'Total: {len(file_paths)} files')

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
        hidden_dim=128
    )
    print('Model loaded!')

    history = train_model(
        model=model,
        train_files=train_files,
        train_labels=train_labels,
        val_files=val_files,
        val_labels=val_labels,
        epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        device=device
    )

    # Save trained model
    save_model(model, 'speech_noise_vi_classifier.pt')
    print('Training complete! Model saved to speech_noise_classifier.pt')


if __name__ == '__main__':
    main()
