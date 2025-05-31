import os
import torch
import numpy as np
from tqdm import tqdm
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from layers import TacotronSTFT  # Ensure this import works in your environment
from utils import load_wav_to_torch  # Ensure this import works in your environment
from hparams import create_hparams  # Adjust if you import this differently

# ==========================
# 🎧 Mel Spectrogram Generator
# ==========================
def make_mel_spectrograms(input_folder: str, output_folder: str) -> None:
    """
    Generates mel spectrograms using TacotronSTFT and saves them as .npy files.
    Utilizes GPU if available.

    Args:
        input_folder (str): Directory containing .wav files.
        output_folder (str): Directory to save .npy mel spectrograms.
    """
    print(f"📁 Generating Mel Spectrograms in: {output_folder}")
    hparams = create_hparams()
    os.makedirs(output_folder, exist_ok=True)

    # זיהוי מכשיר
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # יצירת STFT על ה-device הנבחר
    stft = TacotronSTFT(
        filter_length=hparams.filter_length,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        n_mel_channels=hparams.n_mel_channels,
        sampling_rate=hparams.sampling_rate,
        mel_fmin=hparams.mel_fmin,
        mel_fmax=hparams.mel_fmax
    ).to(device)

    for filename in tqdm(os.listdir(input_folder), desc="🔊 Processing WAVs"):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(input_folder, filename)

        try:
            audio, sampling_rate = load_wav_to_torch(filepath)
            if sampling_rate != stft.sampling_rate:
                raise ValueError(f"{filename}: Expected {stft.sampling_rate}, got {sampling_rate}")

            audio_norm = audio / hparams.max_wav_value
            audio_norm = audio_norm.unsqueeze(0).to(device)

            with torch.no_grad():
                melspec = stft.mel_spectrogram(audio_norm)
                melspec = torch.squeeze(melspec, 0).cpu().numpy()

            out_path = os.path.join(output_folder, filename.replace(".wav", ".npy"))
            np.save(out_path, melspec)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✅ Mel spectrograms saved to: {output_folder}")


# ==========================
# 🚀 Run for Multiple Sets
# ==========================
if __name__ == "__main__":
    data_sets = [
        {
            "input": "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_automatic_data/wavs",
            "output": "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_automatic_data/mel_spectrograms/"
        },
        {
            "input": "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_gold_standard/wavs",
            "output": "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_gold_standard/mel_spectrograms/"
        }
    ]

    for ds in data_sets:
        make_mel_spectrograms(ds["input"], ds["output"])
