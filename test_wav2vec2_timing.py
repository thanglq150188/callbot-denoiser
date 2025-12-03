"""Script to test wav2vec2 classifier on VAD analysis files and measure timing."""

import os
import time
import torch
from wav2vec2_classifier import Wav2Vec2MLPClassifier, load_audio


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model once
    print('Loading Wav2Vec2 model...')
    start_load = time.time()
    model = Wav2Vec2MLPClassifier(freeze_wav2vec2=True)
    model.to(device)
    model.eval()
    
    # Warmup: run a dummy inference to trigger JIT compilation
    print('Warming up model...')
    dummy_waveform = torch.randn(16000).to(device)  # 1 second of random audio
    with torch.no_grad():
        _ = model(dummy_waveform, 16000)
    
    print(f'Model loaded and warmed up in {time.time() - start_load:.2f}s')

    # Get wav files
    wav_dir = 'output/vad_analysis'
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    print(f'\nFound {len(wav_files)} wav files\n')

    # Classify each file
    times = []
    results = []

    for wav_file in sorted(wav_files):
        path = os.path.join(wav_dir, wav_file)
        
        # Load audio
        waveform, sr = load_audio(path)
        waveform = waveform.to(device)
        
        # Time inference
        start = time.time()
        pred_class, confidence = model.predict(waveform, sr)
        elapsed = time.time() - start
        times.append(elapsed)
        
        label = 'speech' if pred_class == 1 else 'noise'
        results.append((wav_file, label, confidence, elapsed))
        print(f'{wav_file}: {label} ({confidence:.2%}) - {elapsed*1000:.1f}ms')

    print(f'\n' + '='*60)
    print(f'Total files: {len(times)}')
    print(f'Avg inference time: {sum(times)/len(times)*1000:.1f}ms')
    print(f'Min inference time: {min(times)*1000:.1f}ms')
    print(f'Max inference time: {max(times)*1000:.1f}ms')
    print(f'Total time: {sum(times)*1000:.1f}ms')


if __name__ == "__main__":
    main()
