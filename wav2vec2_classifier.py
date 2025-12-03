"""
Wav2Vec2 + MLP Classifier for Speech vs Noise Classification

This module provides a simple implementation of using Wav2Vec2 as a feature extractor
with an MLP classifier to classify audio files as either speech or noise.
"""

import os
import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Tuple, Optional


class Wav2Vec2MLPClassifier(nn.Module):
    """
    A classifier that uses Wav2Vec2 as a feature extractor and an MLP for classification.
    
    The model extracts features from audio using the pretrained Wav2Vec2 model,
    then passes them through an MLP to classify as speech (1) or noise (0).
    """
    
    def __init__(
        self,
        wav2vec2_model_name: str = "facebook/wav2vec2-base",
        hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_wav2vec2: bool = True
    ):
        """
        Initialize the classifier.
        
        Args:
            wav2vec2_model_name: Name of the pretrained Wav2Vec2 model
            hidden_dim: Hidden dimension for the MLP
            dropout: Dropout rate
            freeze_wav2vec2: Whether to freeze Wav2Vec2 weights
        """
        super().__init__()
        
        # Load pretrained Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        
        # Freeze Wav2Vec2 if specified
        if freeze_wav2vec2:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Wav2Vec2-base outputs 768-dim features
        wav2vec2_output_dim = self.wav2vec2.config.hidden_size
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(wav2vec2_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 2 classes: noise, speech
        )
    
    def extract_features(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from audio waveform using Wav2Vec2.
        
        Args:
            waveform: Audio waveform tensor of shape (batch, samples) or (samples,)
            sample_rate: Sample rate of the audio (should be 16000 for Wav2Vec2)
            
        Returns:
            Pooled feature tensor of shape (batch, hidden_size)
        """
        # Ensure waveform is 2D (batch, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if necessary (done in load_audio, but check anyway)
        # Note: resampling should be done before calling this method
        
        # Get Wav2Vec2 outputs
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.wav2vec2(waveform)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Mean pooling over time dimension
        pooled_features = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        return pooled_features
    
    def forward(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Logits for classification (batch, 2)
        """
        features = self.extract_features(waveform, sample_rate)
        logits = self.classifier(features)
        return logits
    
    def predict(self, waveform: torch.Tensor, sample_rate: int = 16000) -> Tuple[int, float]:
        """
        Predict whether audio is speech or noise.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (predicted_class, confidence)
            - predicted_class: 0 for noise, 1 for speech
            - confidence: Probability of the predicted class
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(waveform, sample_rate)
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_class].item()
        
        return predicted_class, confidence


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using scipy.
    
    Args:
        waveform: Audio waveform as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform
    
    from scipy import signal
    duration = len(waveform) / orig_sr
    target_length = int(duration * target_sr)
    resampled = signal.resample(waveform, target_length)
    return resampled


def load_audio(file_path: str, target_sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file (supports wav and other formats via soundfile).
    
    Args:
        file_path: Path to the audio file
        target_sample_rate: Target sample rate for resampling
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    # Load audio using soundfile
    waveform, sample_rate = sf.read(file_path)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    
    # Convert to torch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)
    
    return waveform, sample_rate


def classify_audio(
    file_path: str,
    model: Optional[Wav2Vec2MLPClassifier] = None,
    device: str = "cpu"
) -> dict:
    """
    Classify an audio file as speech or noise.
    
    Args:
        file_path: Path to the audio file (mp3 or wav)
        model: Pre-loaded model (will create new one if None)
        device: Device to run inference on
        
    Returns:
        Dictionary with classification results
    """
    # Load or create model
    if model is None:
        model = Wav2Vec2MLPClassifier()
        model.to(device)
    
    # Load audio
    waveform, sample_rate = load_audio(file_path)
    waveform = waveform.to(device)
    
    # Predict
    predicted_class, confidence = model.predict(waveform, sample_rate)
    
    # Map class to label
    label = "speech" if predicted_class == 1 else "noise"
    
    return {
        "file": file_path,
        "label": label,
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "noise": 1 - confidence if predicted_class == 1 else confidence,
            "speech": confidence if predicted_class == 1 else 1 - confidence
        }
    }


# ============================================================================
# Training utilities
# ============================================================================

class AudioDataset(torch.utils.data.Dataset):
    """Simple dataset for audio classification."""
    
    def __init__(self, file_paths: list, labels: list, target_sample_rate: int = 16000):
        """
        Initialize dataset.
        
        Args:
            file_paths: List of paths to audio files
            labels: List of labels (0 for noise, 1 for speech)
            target_sample_rate: Target sample rate
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        waveform, _ = load_audio(self.file_paths[idx], self.target_sample_rate)
        label = self.labels[idx]
        return waveform, label


def collate_fn(batch):
    """Collate function for variable length audio."""
    waveforms, labels = zip(*batch)
    
    # Pad waveforms to same length
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.shape[0] < max_len:
            padding = torch.zeros(max_len - w.shape[0])
            w = torch.cat([w, padding])
        padded_waveforms.append(w)
    
    waveforms = torch.stack(padded_waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return waveforms, labels


def train_model(
    model: Wav2Vec2MLPClassifier,
    train_files: list,
    train_labels: list,
    val_files: list = None,
    val_labels: list = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cpu"
) -> dict:
    """
    Train the classifier.
    
    Args:
        model: The classifier model
        train_files: List of training audio file paths
        train_labels: List of training labels (0=noise, 1=speech)
        val_files: Optional validation file paths
        val_labels: Optional validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Training history dictionary
    """
    model.to(device)
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    if val_files and val_labels:
        val_dataset = AudioDataset(val_files, val_labels)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
    else:
        val_loader = None
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{train_loss / (pbar.n + 1):.4f}',
                'acc': f'{train_correct / train_total:.4f}'
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for waveforms, labels in val_loader:
                    waveforms = waveforms.to(device)
                    labels = labels.to(device)
                    
                    logits = model(waveforms)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    return history


def save_model(model: Wav2Vec2MLPClassifier, path: str):
    """Save model weights."""
    torch.save({
        "classifier_state_dict": model.classifier.state_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_model_weights(model: Wav2Vec2MLPClassifier, path: str):
    """Load model weights."""
    checkpoint = torch.load(path, map_location="cpu")
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    print(f"Model loaded from {path}")
    return model


# ============================================================================
# Demo / Example usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify audio as speech or noise")
    parser.add_argument("--audio", type=str, help="Path to audio file (mp3 or wav)")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.demo:
        print("\n=== Demo: Creating model and classifying synthetic audio ===\n")
        
        # Create model
        print("Loading Wav2Vec2 model...")
        model = Wav2Vec2MLPClassifier(
            wav2vec2_model_name="facebook/wav2vec2-base",
            hidden_dim=256,
            dropout=0.3,
            freeze_wav2vec2=True
        )
        model.to(device)
        print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Create synthetic audio (random noise)
        print("\nClassifying synthetic noise (random signal)...")
        noise_waveform = torch.randn(16000 * 3)  # 3 seconds of noise
        noise_waveform = noise_waveform.to(device)
        
        pred_class, confidence = model.predict(noise_waveform, 16000)
        label = "speech" if pred_class == 1 else "noise"
        print(f"Prediction: {label} (confidence: {confidence:.4f})")
        print("Note: Model is untrained, predictions are random!")
        
        # Create synthetic speech-like signal (sine wave)
        print("\nClassifying synthetic speech-like signal (sine wave)...")
        t = torch.linspace(0, 3, 16000 * 3)
        speech_like = torch.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz sine wave
        speech_like = speech_like.to(device)
        
        pred_class, confidence = model.predict(speech_like, 16000)
        label = "speech" if pred_class == 1 else "noise"
        print(f"Prediction: {label} (confidence: {confidence:.4f})")
        
        print("\n=== Demo complete ===")
        print("To train on real data, use the train_model() function with your dataset.")
    
    elif args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: File not found: {args.audio}")
            exit(1)
        
        print(f"\nClassifying: {args.audio}")
        print("Loading model...")
        
        result = classify_audio(args.audio, device=device)
        
        print(f"\n{'='*50}")
        print(f"File: {result['file']}")
        print(f"Prediction: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        print(f"  - Noise: {result['probabilities']['noise']:.2%}")
        print(f"  - Speech: {result['probabilities']['speech']:.2%}")
        print(f"{'='*50}")
        print("\nNote: Model is untrained! For accurate predictions, train on labeled data.")
    
    else:
        print("Usage:")
        print("  python wav2vec2_classifier.py --demo           # Run demo with synthetic audio")
        print("  python wav2vec2_classifier.py --audio <path>   # Classify an audio file")
        print("\nExample:")
        print("  python wav2vec2_classifier.py --audio sample.wav")
        print("  python wav2vec2_classifier.py --audio sample.mp3")
