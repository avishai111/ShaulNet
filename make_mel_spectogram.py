import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
from tqdm import tqdm
import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm

import os
import torch
import numpy as np
from tqdm import tqdm
from layers import TacotronSTFT  # ודא שזה מיובא נכון
from utils import load_wav_to_torch  # ודא שזה מיובא נכון
from hparams import create_hparams  # או מייבוא מתאים לפרויקט שלך

def make_mel_spectrograms(input_folder, output_folder):
    print("Generating Mel Spectrograms with TacotronSTFT...")
    hparams = create_hparams()
    os.makedirs(output_folder, exist_ok=True)

    stft = TacotronSTFT(
        filter_length=hparams.filter_length,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        n_mel_channels=hparams.n_mel_channels,
        sampling_rate=hparams.sampling_rate,
        mel_fmin=hparams.mel_fmin,
        mel_fmax=hparams.mel_fmax
    )

    for filename in tqdm(os.listdir(input_folder), desc="Processing files"):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)

            try:
                # Load and normalize audio
                audio, sampling_rate = load_wav_to_torch(filepath)
                if sampling_rate != stft.sampling_rate:
                    raise ValueError(f"{filename}: Sampling rate {sampling_rate} doesn't match target {stft.sampling_rate}")

                audio_norm = audio / hparams.max_wav_value
                audio_norm = audio_norm.unsqueeze(0)

                with torch.no_grad():
                    melspec = stft.mel_spectrogram(audio_norm)
                    melspec = torch.squeeze(melspec, 0).cpu().numpy()

                # Save .npy file
                output_path = os.path.join(output_folder, filename.replace(".wav", ".npy"))
                np.save(output_path, melspec)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Mel spectrogram generation complete.")




# Input and output folder paths
input_folder = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_automatic_data/wavs/"   # Change this to your actual input directory
output_folder =  "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_automatic_data/mel_spectrograms/"   # Change this to your actual output directory

make_mel_spectrograms(input_folder, output_folder)

# Input and output folder paths
input_folder = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_gold_standard/wavs/"   # Change this to your actual input directory
output_folder =  "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_gold_standard/mel_spectrograms/"   # Change this to your actual output directory

make_mel_spectrograms(input_folder, output_folder)


# def create_mels():
#     print("Generating Mels")
#     stft = layers.TacotronSTFT(
#                 hparams.filter_length, hparams.hop_length, hparams.win_length,
#                 hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
#                 hparams.mel_fmax)
#     def save_mel(filename):
#         audio, sampling_rate = load_wav_to_torch(filename)
#         if sampling_rate != stft.sampling_rate:
#             raise ValueError("{} {} SR doesn't match target {} SR".format(filename, 
#                 sampling_rate, stft.sampling_rate))
#         audio_norm = audio / hparams.max_wav_value
#         audio_norm = audio_norm.unsqueeze(0)
#         audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
#         melspec = stft.mel_spectrogram(audio_norm)
#         melspec = torch.squeeze(melspec, 0).cpu().numpy()
#         np.save(filename.replace('.wav', ''), melspec)


# import os
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display

# def plot_random_mel(mel_folder: str, save_folder: str = None):
#     mel_files = [f for f in os.listdir(mel_folder) if f.endswith('.npy')]
#     if not mel_files:
#         print("No .npy files found in the directory.")
#         return

#     random_file = random.choice(mel_files)
#     mel_path = os.path.join(mel_folder, random_file)
#     mel = np.load(mel_path)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mel, x_axis='time', y_axis='mel', cmap='magma')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(f"Mel Spectrogram: {random_file}")
#     plt.tight_layout()

#     # Save to file instead of showing
#     if save_folder is None:
#         save_folder = os.path.join("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/", "mel_plots")
#     os.makedirs(save_folder, exist_ok=True)

#     save_path = os.path.join(save_folder, f"{os.path.splitext(random_file)[0]}_plot.png")
#     plt.savefig(save_path)
#     plt.close()

#     print(f"Saved plot to: {save_path}")

# # Example usage
# plot_random_mel("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_automatic_data/mel_spectrograms/")
