"""
VoiceMD Model Downloader
Downloads models from GitHub Releases
"""

import os
import sys
import requests
from pathlib import Path
from urllib.request import urlretrieve


# GitHub Release URLs
GITHUB_USER = "Honey181"
GITHUB_REPO = "voicemd"
RELEASE_TAG = "v1.0.0"

MODELS = {
    "best_model (Small dataset).pt": {
        "url": f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/best_model_small.pt",
        "size_mb": 2.2
    },
    "best_model_commonvoice (Commonvoice).pt": {
        "url": f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/best_model_commonvoice.pt",
        "size_mb": 2.2
    }
}


def get_models_dir():
    """Get or create models directory"""
    # Check if running from source
    if Path('voicemd').exists():
        models_dir = Path('.')
    else:
        # Installed via pip - use user's home
        models_dir = Path.home() / '.voicemd' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_file(url, dest_path, desc="Downloading"):
    """Download file with progress"""
    print(f"{desc}...", end='', flush=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(f"\r{desc}... {progress:.1f}%", end='', flush=True)
        
        print(f"\r{desc}... Done! ✓")
        return True
    except Exception as e:
        print(f"\r{desc}... Failed: {e}")
        return False


def check_models():
    """Check if models exist"""
    models_dir = get_models_dir()
    missing = []
    
    for model_file in MODELS.keys():
        if not (models_dir / model_file).exists():
            missing.append(model_file)
    
    return missing


def download_models(force=False):
    """Download missing models"""
    models_dir = get_models_dir()
    
    print("VoiceMD Model Downloader")
    print("=" * 50)
    print(f"Models directory: {models_dir}")
    print()
    
    if not force:
        missing = check_models()
        if not missing:
            print("✓ All models already downloaded!")
            return True
        print(f"Downloading {len(missing)} model(s)...")
    else:
        print("Force downloading all models...")
        missing = list(MODELS.keys())
    
    print()
    success = True
    
    for model_file in missing:
        model_info = MODELS[model_file]
        dest_path = models_dir / model_file
        
        print(f"Model: {model_file} ({model_info['size_mb']} MB)")
        if download_file(model_info['url'], dest_path, "  Downloading"):
            print(f"  Saved to: {dest_path}")
        else:
            success = False
            print(f"  Failed to download!")
        print()
    
    if success:
        print("=" * 50)
        print("✓ All models downloaded successfully!")
        print()
        print("You can now run: voicemd-gui")
    else:
        print("=" * 50)
        print("✗ Some downloads failed. Please try again.")
    
    return success


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download VoiceMD models')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download all models')
    parser.add_argument('--check', action='store_true',
                       help='Only check if models exist')
    
    args = parser.parse_args()
    
    if args.check:
        missing = check_models()
        if missing:
            print(f"Missing models: {', '.join(missing)}")
            sys.exit(1)
        else:
            print("✓ All models present")
            sys.exit(0)
    
    success = download_models(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

