import torch
import torchaudio
import librosa
import numpy as np


def compute_specgram(waveform, sr, spec_type, normalize):

    if spec_type == 'librosa_melspec':
        specgram = librosa.feature.melspectrogram(y=waveform, sr=sr, hop_length=512, win_length=512, fmax=8000, n_mels=80)
        specgram = librosa.power_to_db(specgram, ref=np.max)
        specgram = torch.tensor(specgram).unsqueeze(dim=0)

    elif spec_type == 'pytorch_spec':
        specgram = torchaudio.transforms.Spectrogram(n_fft=400, normalized=True)(waveform)

    elif spec_type == 'pytorch_melspec':
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=40)(waveform)

    elif spec_type == 'pytorch_mfcc':
        specgram = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40, log_mels=False, melkwargs={'n_fft': 800})(waveform)

    else:
        raise ValueError("spec_type not defined")

    if normalize:
        # Center values about 0 since
        # dbs range from -80 to 0
        specgram = (specgram + 40)

    return specgram


def load_waveform(fname):
    # Load audio using soundfile backend (works without FFmpeg)
    import soundfile as sf
    
    try:
        # Use soundfile directly - most reliable, works without FFmpeg
        waveform, sr = sf.read(fname, always_2d=False)
        
        # Convert to mono if stereo
        if waveform.ndim > 1:
            # Average channels
            waveform = np.mean(waveform, axis=1)
            
    except Exception as e:
        # If soundfile fails, try librosa as last resort
        try:
            waveform, sr = librosa.load(fname, sr=None, mono=True)
        except Exception as e2:
            raise RuntimeError(f"Could not load audio file {fname}. "
                             f"Soundfile error: {e}. Librosa error: {e2}")
    
    # loop the sound to make the segment long enough
    min_seconds = 5
    while (len(waveform) / sr) < min_seconds:
        waveform = np.concatenate((waveform, waveform))

    return waveform, sr
