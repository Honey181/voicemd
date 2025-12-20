# VoiceMD - Voice Analysis Application

A modern, offline voice analysis tool that predicts speaker characteristics based on acoustic features.

## Features

- 🎨 **Modern Interface** - Clean, intuitive GUI with modern design
- 🔄 **Multi-Model Support** - Switch between different trained models in real-time
- 💾 **Fully Offline** - No internet required after initial setup
- 🎤 **Multiple Formats** - Supports WAV, MP3, OGG, FLAC, M4A
- 📦 **Easy Installation** - Simple pip install, models auto-download

## Installation

### Requirements

- **Python 3.8 - 3.13** (Python 3.14+ not yet supported - dependencies `numba`/`llvmlite` don't support 3.14. Use Python 3.13 or earlier)
- ~500 MB disk space (including models and dependencies)
- Internet connection for first-time model download only

### Quick Install (Recommended - No Git Required!)

```bash
pip install voicemd-gui
```

On some Linux distributions, eg. Arch Linux, you might need to use `pipx` instead of `pip`.

Then run:

```bash
voicemd-gui
```

Models (~4.4 MB) will automatically download from GitHub Releases on first launch.

### Alternative Install (From GitHub)

If you want the latest development version:

```bash
pip install git+https://github.com/Honey181/voicemd.git
```

### Manual Install (From Source)

1. Download the repository as ZIP from GitHub
2. Extract it
3. Open terminal in the extracted folder
4. Run: `pip install .`
5. Run: `voicemd-gui`

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

### Remove the Application

```bash
pip uninstall voicemd-gui -y
```

### Remove Downloaded Models (Optional)

The model files (~4.4 MB) are stored separately and will remain after uninstalling. To completely remove them:

**Windows:**
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.voicemd"
```

**macOS/Linux:**
```bash
rm -rf ~/.voicemd
```

**Location of models:**
- When installed via pip: `~/.voicemd/models/` (or `%USERPROFILE%\.voicemd\models\` on Windows)
- When running from source: Project root directory

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
- **Linux:**
  - **Debian-based (Ubuntu, Linux Mint)**: `sudo apt install ffmpeg`
  - **Fedora**: `sudo dnf install ffmpeg-free`
  - **Arch-based (Arch Linux, EndeavourOS, Manjaro)**: `sudo pacman -S ffmpeg`
  - **openSUSE**: `sudo zypper install ffmpeg-4`
  - **NixOS**: `sudo nix-env -iA nixos.ffmpeg`

### LLVM Version Compatibility (Linux)

If you get an error like `llvmlite only officially supports LLVM 20` during installation:

**Option 1 - Use Conda (Recommended):**
```bash
conda create -n voicemd python=3.10
conda activate voicemd
conda install -c conda-forge llvmlite numba
pip install git+https://github.com/Honey181/voicemd.git
```

**Option 2 - Install specific LLVM version:**
```bash
# Ubuntu/Debian
sudo apt install llvm-14 llvm-14-dev
export LLVM_CONFIG=/usr/bin/llvm-config-14
pip install git+https://github.com/Honey181/voicemd.git
```

**Option 3 - Use older Python:**
```bash
# Python 3.9 or 3.10 have better llvmlite compatibility
python3.10 -m pip install git+https://github.com/Honey181/voicemd.git
```

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
