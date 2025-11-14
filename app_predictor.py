"""
VoiceMD Prediction Engine
Handles model loading and audio analysis for offline use.
"""

import torch
import torchaudio
import librosa
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple
import warnings
import sys
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Import model components
from voicemd.models.model_loader import load_model

# Default configuration (embedded to avoid file dependency issues)
DEFAULT_CONFIG = {
    'architecture': 'longfilter',
    'in_channels': 1,
    'spec_type': 'librosa_melspec',
    'normalize_spectrums': True,
    'window_len': 256,
    'batch_size': 1,
    'pretrained': True,
    'seed': 42
}
from voicemd.data.process_sound import load_waveform, compute_specgram


class VoiceAnalyzer:
    """Handles voice analysis using pre-trained models"""
    
    def __init__(self, config_path: str = None, model_path: str = None):
        """
        Initialize the voice analyzer
        
        Args:
            config_path: Path to config.yaml file (optional - uses default if not found)
            model_path: Path to trained model weights
        """
        # Load configuration
        if config_path is None:
            try:
                config_path = self._find_config()
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f"Loaded configuration from: {config_path}")
            except FileNotFoundError:
                # Use embedded default configuration
                self.config = DEFAULT_CONFIG.copy()
                config_path = "embedded"
                print("Using embedded default configuration")
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded configuration from: {config_path}")
        
        self.config_path = config_path
        self.current_model_path = model_path
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model architecture (without weights)
        print("Initializing model architecture...")
        self.model = load_model(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Load weights if model path provided
        if model_path:
            self.load_model_weights(model_path)
        
        print("Analyzer ready!")
    
    def load_model_weights(self, model_path: str) -> bool:
        """
        Load or switch model weights
        
        Args:
            model_path: Path to model weights file
            
        Returns:
            True if successful, False otherwise
        """
        if not Path(model_path).exists():
            print(f"Warning: Model weights not found at {model_path}")
            return False
        
        try:
            print(f"Loading model weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.current_model_path = model_path
            print("Model weights loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return False
    
    def _find_config(self) -> str:
        """Find configuration file"""
        # Try relative to current directory first (for local development)
        local_paths = [
            'app_config.yaml',
            'voicemd/config.yaml',
            'config.yaml'
        ]
        
        for path in local_paths:
            if Path(path).exists():
                return path
        
        # Try relative to this file's directory (for installed package)
        package_dir = Path(__file__).parent
        package_paths = [
            package_dir / 'app_config.yaml',
            package_dir / '..' / 'app_config.yaml',
            package_dir / 'config.yaml',
        ]
        
        for path in package_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError("Could not find configuration file")
    
    def _find_model(self) -> str:
        """Find model weights file"""
        possible_paths = [
            'best_model.pt',
            'best_model (Small dataset).pt',
            'best_model_commonvoice (Commonvoice).pt',
            'voicemd/output/best_model.pt',
            'output/best_model.pt',
            'model.pt'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # Return default path even if not found
        return 'best_model.pt'
    
    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available models
        
        Returns:
            List of dictionaries with model info
        """
        # Model metadata (embedded as fallback)
        MODEL_DESCRIPTIONS = {
            'best_model (Small dataset).pt': {
                'name': 'Small Dataset Model',
                'description': 'Trained on small dataset - faster, good for general use'
            },
            'best_model_commonvoice (Commonvoice).pt': {
                'name': 'CommonVoice Model',
                'description': 'Trained on CommonVoice dataset - more robust, diverse accents'
            }
        }
        
        # Determine search paths
        search_paths = []
        
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            search_paths.append(Path(sys._MEIPASS))
        else:
            # Check if running from source directory
            is_source = (Path('setup.py').exists() or 
                        Path('pyproject.toml').exists() or
                        (Path('voicemd').is_dir() and Path('app_gui.py').exists()))
            
            if is_source:
                # Running from source - check current directory
                search_paths.append(Path('.'))
            else:
                # Installed via pip - check user home directory first
                home_models = Path.home() / '.voicemd' / 'models'
                if home_models.exists():
                    search_paths.append(home_models)
                # Also check current directory as fallback
                search_paths.append(Path('.'))
        
        models = []
        found_files = set()  # Track found files to avoid duplicates (use absolute paths)
        
        # Try each search path
        for base_path in search_paths:
            # Try to load from config file first
            models_config_path = base_path / 'models_config.yaml'
            if models_config_path.exists():
                try:
                    with open(models_config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        for model_info in config.get('models', []):
                            model_path = base_path / model_info['file']
                            if model_path.exists():
                                # Use absolute path for deduplication
                                abs_path = str(model_path.resolve())
                                if abs_path not in found_files:
                                    found_files.add(abs_path)
                                    models.append({
                                        'name': model_info['name'],
                                        'path': str(model_path),
                                        'description': model_info.get('description', ''),
                                        'size_mb': model_path.stat().st_size / (1024 * 1024)
                                    })
                except Exception as e:
                    print(f"Error loading models config from {models_config_path}: {e}")
            
            # Search for model files directly
            for filename, metadata in MODEL_DESCRIPTIONS.items():
                full_path = base_path / filename
                if full_path.exists():
                    # Use absolute path for deduplication
                    abs_path = str(full_path.resolve())
                    if abs_path not in found_files:
                        found_files.add(abs_path)
                        models.append({
                            'name': metadata['name'],
                            'path': str(full_path),
                            'description': metadata['description'],
                            'size_mb': full_path.stat().st_size / (1024 * 1024)
                        })
        
        # If no models found with descriptions, try generic search
        if not models:
            for base_path in search_paths:
                fallback_paths = [
                    ('best_model (Small dataset).pt', 'Small Dataset Model'),
                    ('best_model_commonvoice (Commonvoice).pt', 'CommonVoice Model'),
                    ('best_model.pt', 'Default Model'),
                    ('model.pt', 'Model'),
                ]
                
                for filename, name in fallback_paths:
                    full_path = base_path / filename
                    if full_path.exists():
                        # Use absolute path for deduplication
                        abs_path = str(full_path.resolve())
                        if abs_path not in found_files:
                            found_files.add(abs_path)
                            models.append({
                                'name': name,
                                'path': str(full_path),
                                'description': '',
                                'size_mb': full_path.stat().st_size / (1024 * 1024)
                            })
        
        return models
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # Load waveform
        waveform, sr = load_waveform(audio_path)
        
        # Compute spectrogram
        spec_type = self.config.get('spec_type', 'librosa_melspec')
        normalize = self.config.get('normalize_spectrums', True)
        
        specgram = compute_specgram(waveform, sr, spec_type, normalize)
        
        return specgram
    
    def _create_windows(self, specgram: torch.Tensor, window_len: int = 256) -> torch.Tensor:
        """
        Create sliding windows from spectrogram
        
        Args:
            specgram: Input spectrogram
            window_len: Length of each window
            
        Returns:
            Tensor of windowed spectrograms
        """
        # Get window length from config
        window_len = self.config.get('window_len', 256)
        
        # specgram shape: [1, freq_bins, time_steps]
        _, freq_bins, time_steps = specgram.shape
        
        windows = []
        
        # Create overlapping windows
        step_size = window_len // 2  # 50% overlap
        
        for start in range(0, time_steps - window_len + 1, step_size):
            end = start + window_len
            window = specgram[:, :, start:end]
            windows.append(window)
        
        # If no complete windows, use the whole spectrogram with padding
        if len(windows) == 0:
            if time_steps < window_len:
                # Pad to window_len
                pad_len = window_len - time_steps
                window = torch.nn.functional.pad(specgram, (0, pad_len))
                windows.append(window)
            else:
                windows.append(specgram[:, :, :window_len])
        
        return torch.stack(windows)
    
    def analyze(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze audio file and return predictions
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing: {audio_path}")
        
        # Preprocess audio
        specgram = self._preprocess_audio(audio_path)
        
        # Create windows
        windows = self._create_windows(specgram)
        
        # Move to device
        windows = windows.to(self.device)
        
        # Run inference
        all_probs = []
        
        with torch.no_grad():
            for window in windows:
                # Add batch dimension if needed
                if window.dim() == 3:
                    window = window.unsqueeze(0)
                
                # Get prediction
                output = self.model(window)
                probs = torch.nn.functional.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Average probabilities across all windows
        all_probs = np.concatenate(all_probs, axis=0)
        avg_prob = np.mean(all_probs, axis=0)
        
        # Results (class 0: female, class 1: male)
        female_prob = avg_prob[0] * 100
        male_prob = avg_prob[1] * 100
        
        prediction = "Male" if male_prob > female_prob else "Female"
        confidence = max(male_prob, female_prob)
        
        results = {
            'prediction': prediction,
            'confidence': confidence,
            'male_probability': male_prob,
            'female_probability': female_prob
        }
        
        print(f"Prediction: {prediction} ({confidence:.1f}% confidence)")
        print(f"Male: {male_prob:.2f}% | Female: {female_prob:.2f}%")
        
        return results


def test_analyzer():
    """Test the analyzer with a sample file"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python app_predictor.py <audio_file>")
        return
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        return
    
    analyzer = VoiceAnalyzer()
    results = analyzer.analyze(audio_file)
    
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Prediction: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.1f}%")
    print(f"Male Probability: {results['male_probability']:.2f}%")
    print(f"Female Probability: {results['female_probability']:.2f}%")
    print("="*50)


if __name__ == "__main__":
    test_analyzer()

