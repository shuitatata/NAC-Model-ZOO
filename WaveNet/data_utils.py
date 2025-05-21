import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import glob
from torch.utils.data import Dataset

# --- Helper Functions: µ-law encoding/decoding & one-hot ---
def mu_law_encode(audio, quantization_channels=256):
    """
    Encodes audio signal using µ-law algorithm.
    Args:
        audio (np.ndarray): Input audio signal, expected to be in range [-1, 1].
        quantization_channels (int): Number of discrete quantization levels.
    Returns:
        np.ndarray: µ-law encoded audio signal (quantized integer values).
    """
    mu = float(quantization_channels - 1)
    signal = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    # Scale to [0, mu] and quantize
    signal = (signal + 1) / 2 * mu + 0.5
    return signal.astype(np.int64)

def mu_law_decode(output, quantization_channels=256):
    """
    Decodes µ-law encoded audio signal.
    Args:
        output (np.ndarray): µ-law encoded signal (quantized integer values).
        quantization_channels (int): Number of discrete quantization levels.
    Returns:
        np.ndarray: Decoded audio signal in range [-1, 1].
    """
    mu = float(quantization_channels - 1)
    signal = (output.astype(np.float32) / mu) * 2.0 - 1.0
    signal = np.sign(signal) * (np.exp(np.abs(signal) * np.log1p(mu)) - 1.0) / mu
    return signal

def one_hot_encode(indices, num_classes):
    """
    Converts integer indices to one-hot encoded vectors.
    Args:
        indices (torch.Tensor): Tensor of indices (B, L).
        num_classes (int): Number of classes for one-hot encoding.
    Returns:
        torch.Tensor: One-hot encoded tensor (B, L, num_classes).
    """
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Dataset Class ---
class VCTKSpeakerDataset(Dataset):
    def __init__(self, audio_dir, segment_length, quantization_channels=256, target_sample_rate=16000):
        """
        Args:
            audio_dir (str): Path to the directory containing .flac or .wav files for a specific speaker.
            segment_length (int): Length of the audio segments to be used for training.
            quantization_channels (int): Number of quantization channels for µ-law encoding.
            target_sample_rate (int): Sample rate to resample audio to.
        """
        super().__init__()
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        self.quantization_channels = quantization_channels
        self.target_sample_rate = target_sample_rate
        
        self.audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
        if not self.audio_files:
             self.audio_files = glob.glob(os.path.join(audio_dir, '*.flac'))
        
        if not self.audio_files:
            raise FileNotFoundError(f"No .wav or .flac files found in {audio_dir}")

        print(f"Found {len(self.audio_files)} audio files in {audio_dir}")
        self.resampler = None
        self.all_audio_data = self._load_and_preprocess_all_audio()

    def _load_and_preprocess_all_audio(self):
        all_processed_audio = []
        for audio_file in self.audio_files:
            try:
                waveform, sample_rate = torchaudio.load(audio_file)
                
                if sample_rate != self.target_sample_rate:
                    if self.resampler is None or self.resampler.orig_freq != sample_rate:
                         self.resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                    waveform = self.resampler(waveform)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                waveform = waveform.squeeze().numpy()
                
                encoded_waveform = mu_law_encode(waveform, self.quantization_channels)
                all_processed_audio.append(torch.from_numpy(encoded_waveform))

            except Exception as e:
                print(f"Skipping file {audio_file} due to error: {e}")
        
        if not all_processed_audio:
            raise ValueError("No audio files could be processed.")
            
        return torch.cat(all_processed_audio)


    def __len__(self):
        if len(self.all_audio_data) < self.segment_length:
            return 0
        return len(self.all_audio_data) - self.segment_length + 1

    def __getitem__(self, idx):
        segment = self.all_audio_data[idx : idx + self.segment_length]
        return segment 