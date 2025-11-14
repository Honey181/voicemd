# VoiceMD - Voice Analysis Application

A modern, offline voice analysis tool that predicts speaker characteristics based on acoustic features.

## Features

- 🎨 **Modern Interface** - Clean, intuitive GUI with modern design
- 🔄 **Multi-Model Support** - Switch between different trained models in real-time
- 💾 **Fully Offline** - No internet required after initial setup
- 🎤 **Multiple Formats** - Supports WAV, MP3, OGG, FLAC, M4A
- 📦 **Easy Installation** - Simple pip install, models auto-download

## Installation

### Quick Install (Recommended)

```bash
pip install git+https://github.com/Honey181/voicemd.git
```

Then run:

```bash
voicemd-gui
```

Models (~4.4 MB) will automatically download from GitHub Releases on first launch.

### Requirements

- Python 3.8 or higher
- ~500 MB disk space (including models and dependencies)
- Internet connection for first-time model download only

### Manual Installation

If you prefer to run from source:

```bash
# Clone the repository
git clone https://github.com/Honey181/voicemd.git
cd voicemd

# Install dependencies
pip install -r requirements_app.txt

# Run the app
python app_gui.py
```

Models will auto-download on first run, or manually run:
```bash
python download_models.py
```

## Usage

1. Select a model from the dropdown (Small Dataset or CommonVoice)
2. Browse for an audio file (WAV, MP3, OGG, FLAC, M4A)
3. Click "Analyze Voice"
4. View results instantly

## Uninstallation

To completely remove VoiceMD from your system:

```bash
pip uninstall voicemd -y
```

This will remove the application. The downloaded model files (~4.4 MB) remain in your user directory and can be manually deleted if desired.

## Multi-Model Support

VoiceMD includes two trained models that can be switched at runtime:

- **Small Dataset Model** - Faster, good for general use
- **CommonVoice Model** - More robust, handles diverse accents

Switch between models using the dropdown in the app. No restart required!

## Troubleshooting

### Models Won't Download

**Download manually:**
1. Go to https://github.com/Honey181/voicemd/releases
2. Download both `.pt` model files
3. Place them in the project root directory

**Or use the download script:**
```bash
python download_models.py
```

### Import Errors

Update dependencies:
```bash
pip install --upgrade -r requirements_app.txt
```

### Audio Loading Issues

The app uses soundfile/librosa (no FFmpeg required). If you still get errors:
- **Windows:** `choco install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

## Technical Details

- **Framework:** PyTorch for model inference
- **GUI:** Tkinter for cross-platform interface
- **Audio Processing:** librosa, soundfile, torchaudio
- **No FFmpeg Required:** Uses soundfile/librosa for audio loading

## License

MIT License - Copyright (c) 2020, Jeremy Pinto

See `LICENSE` file for full details.

## Credits

**Original Project:** [VoiceMD](https://github.com/jerpint/voicemd) by [@jerpint](https://github.com/jerpint) (Jeremy Pinto)

**Enhanced by:** [@Honey181](https://github.com/Honey181) - Modern UI and easy installation

This project builds upon the excellent work of the original VoiceMD team. All credit for the model architecture, training pipeline, and core functionality goes to them.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
