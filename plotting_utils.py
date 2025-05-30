import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ============================
# 🔄 Utility Functions
# ============================
def save_figure_to_numpy(fig):
    """Converts a Matplotlib figure to a NumPy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


# ============================
# 📊 Visualization Utilities
# ============================
def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return save_and_close(fig)


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    return save_and_close(fig)


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5, color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5, color='red', marker='.', s=1, label='predicted')
    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()
    return save_and_close(fig)


def save_and_close(fig):
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data


def plot_and_save_spectrogram_from_npy(file_path, output_image_path=None, title='Spectrogram', cmap='viridis'):
    """
    Load and plot a spectrogram stored in a .npy file, saving it as an image.
    """
    try:
        spectrogram = np.load(file_path)

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower', cmap=cmap)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar(label='Amplitude')
        plt.tight_layout()

        if output_image_path is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_image_path = f"{base_name}_spectrogram.png"

        plt.savefig(output_image_path, dpi=300)
        plt.close()
        print(f"✅ Spectrogram saved to: {output_image_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


# ============================
# 🚀 Example Usage
# ============================
if __name__ == "__main__":
    import librosa

    audio_path = '/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_automatic_data/wavs/automatic_0000.wav'
    spectrogram_path = '/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/data/saspeech_automatic_data/mel_spectrograms/automatic_0000.npy'

    y, sr = librosa.load(audio_path, sr=None)
    print(f"Sample rate: {sr} Hz")

    plot_and_save_spectrogram_from_npy(spectrogram_path, './automatic_0000_spectrogram.png')
