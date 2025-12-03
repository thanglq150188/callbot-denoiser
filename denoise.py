"""
Direct DeepFilterNet denoising without subprocess.
Supports both file-based and bytes-based processing.
"""
import argparse
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn

from df.enhance import enhance, init_df
from df.io import resample


class DeepFilterNetDenoiser:
    """
    DeepFilterNet denoiser with support for bytes input/output.
    
    Args:
        post_filter: Enable post-filter for extra noise reduction
        atten_lim_db: Noise attenuation limit in dB. None for unlimited.
                      E.g., 12 means only suppress up to 12 dB of noise.
        pad: Compensate for STFT/ISTFT delay (recommended True)
    """
    
    def __init__(
        self,
        post_filter: bool = False,
        atten_lim_db: Optional[float] = None,
        pad: bool = True,
    ):
        self.post_filter = post_filter
        self.atten_lim_db = atten_lim_db
        self.pad = pad
        
        # Initialize model lazily
        self._model: Optional[nn.Module] = None
        self._df_state = None
        self._target_sr: Optional[int] = None
    
    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if self._model is None:
            print("Loading DeepFilterNet model...")
            self._model, self._df_state, _ = init_df(post_filter=self.post_filter)
            self._target_sr = self._df_state.sr()
    
    @property
    def sample_rate(self) -> int:
        """Get the model's expected sample rate (48000 Hz)."""
        self._ensure_model_loaded()
        return self._target_sr
    
    def denoise_tensor(
        self,
        audio: Tensor,
        sample_rate: int,
        atten_lim_db: Optional[float] = None,
    ) -> Tuple[Tensor, int]:
        """
        Denoise audio tensor.
        
        Args:
            audio: Audio tensor of shape [C, T] or [T]
            sample_rate: Sample rate of the input audio
            atten_lim_db: Override default attenuation limit
        
        Returns:
            Tuple of (denoised_audio, output_sample_rate)
        """
        self._ensure_model_loaded()
        
        # Ensure correct shape [C, T]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        # Resample if needed
        orig_sr = sample_rate
        if sample_rate != self._target_sr:
            audio = resample(audio, sample_rate, self._target_sr)
        
        # Apply denoising
        atten = atten_lim_db if atten_lim_db is not None else self.atten_lim_db
        enhanced = enhance(
            self._model,
            self._df_state,
            audio,
            pad=self.pad,
            atten_lim_db=atten,
        )
        
        # Resample back to original sample rate if needed
        if orig_sr != self._target_sr:
            enhanced = resample(enhanced, self._target_sr, orig_sr)
            return enhanced, orig_sr
        
        return enhanced, self._target_sr
    
    def denoise_bytes(
        self,
        audio_bytes: bytes,
        atten_lim_db: Optional[float] = None,
        output_format: str = "WAV",
        output_subtype: str = "PCM_16",
    ) -> bytes:
        """
        Denoise audio from bytes and return bytes.
        
        Args:
            audio_bytes: Input audio as bytes (WAV, FLAC, etc.)
            atten_lim_db: Override default attenuation limit
            output_format: Output format (WAV, FLAC, OGG, etc.)
            output_subtype: Output subtype (PCM_16, PCM_24, FLOAT, etc.)
        
        Returns:
            Denoised audio as bytes
        """
        # Load audio from bytes
        audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        # Convert to tensor [C, T]
        if audio_np.ndim == 1:
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio_np.T)
        
        # Denoise
        enhanced, out_sr = self.denoise_tensor(audio_tensor, sample_rate, atten_lim_db)
        
        # Convert back to numpy [T, C] for soundfile
        enhanced_np = enhanced.numpy()
        if enhanced_np.ndim == 2:
            enhanced_np = enhanced_np.T
        
        # Write to bytes
        output_buffer = io.BytesIO()
        sf.write(output_buffer, enhanced_np, out_sr, format=output_format, subtype=output_subtype)
        output_buffer.seek(0)
        return output_buffer.read()
    
    def denoise_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        atten_lim_db: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Denoise numpy array audio.
        
        Args:
            audio: Audio numpy array of shape [C, T], [T, C], or [T]
            sample_rate: Sample rate of the input audio
            atten_lim_db: Override default attenuation limit
        
        Returns:
            Tuple of (denoised_audio, output_sample_rate)
            Output shape matches input shape.
        """
        # Determine input shape format
        input_1d = audio.ndim == 1
        input_channels_last = False
        
        if audio.ndim == 2:
            # Heuristic: if second dim is 1 or 2, assume [T, C]
            if audio.shape[1] in (1, 2) and audio.shape[0] > 2:
                input_channels_last = True
                audio = audio.T  # [T, C] -> [C, T]
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        
        # Denoise
        enhanced, out_sr = self.denoise_tensor(audio_tensor, sample_rate, atten_lim_db)
        
        # Convert back to numpy
        enhanced_np = enhanced.numpy()
        
        # Match input format
        if input_1d:
            enhanced_np = enhanced_np.squeeze()
        elif input_channels_last:
            enhanced_np = enhanced_np.T  # [C, T] -> [T, C]
        
        return enhanced_np, out_sr
    
    def denoise_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        atten_lim_db: Optional[float] = None,
    ) -> str:
        """
        Denoise audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output file (optional, auto-generated if None)
            atten_lim_db: Override default attenuation limit
        
        Returns:
            Path to output file
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path.with_stem(f"{input_path.stem}_denoised")
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read file as bytes
        with open(input_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Denoise
        enhanced_bytes = self.denoise_bytes(audio_bytes, atten_lim_db)
        
        # Write output
        with open(output_path, 'wb') as f:
            f.write(enhanced_bytes)
        
        return str(output_path)


# Convenience function for simple usage
def denoise_audio(
    input_path: str,
    output_path: Optional[str] = None,
    post_filter: bool = False,
    atten_lim_db: Optional[float] = None,
) -> str:
    """
    Denoise a WAV file using DeepFilterNet.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output file (optional)
        post_filter: Enable post-filter for extra noise reduction
        atten_lim_db: Noise attenuation limit in dB (None for unlimited)
    
    Returns:
        Path to output file
    """
    denoiser = DeepFilterNetDenoiser(post_filter=post_filter, atten_lim_db=atten_lim_db)
    return denoiser.denoise_file(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Denoise WAV files using DeepFilterNet (direct API)"
    )
    parser.add_argument("input", help="Path to input WAV file")
    parser.add_argument("-o", "--output", help="Path to output WAV file (optional)")
    parser.add_argument(
        "--post-filter", "-pf",
        action="store_true",
        help="Enable post-filter for extra noise reduction"
    )
    parser.add_argument(
        "--atten-lim", "-a",
        type=float,
        default=None,
        help="Noise attenuation limit in dB (e.g., 12 means max 12dB reduction)"
    )
    
    args = parser.parse_args()
    
    output = denoise_audio(
        args.input,
        args.output,
        post_filter=args.post_filter,
        atten_lim_db=args.atten_lim,
    )
    print(f"Denoised audio saved to: {output}")


if __name__ == "__main__":
    main()

